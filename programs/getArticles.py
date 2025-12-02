#!/usr/bin/env python3
import asyncio
import aiohttp
import aiofiles
import json
import time
import os
import platform
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from pandas import read_csv
from programs.processes.processHTML import process_html, init_worker, preload_models

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

last_request_claim = {}
## simple html fetch w/ rate limiting
async def fetch_html(url: str, domain: str, session: aiohttp.ClientSession, min_delay: float):
    while True and min_delay>0:
        now = time.monotonic()
        last = last_request_claim.get(domain, 0)

        wait_time = last + min_delay - now
        if wait_time > 0: await asyncio.sleep(wait_time)
        else: break
    
    last_request_claim[domain] = time.monotonic()

    try:
        resp = await session.get(url, timeout=20.0, headers=HEADERS)
    except Exception as e:
        print(f"[ERROR] on session: {e}")
        return e

    try:
        async with resp:
            resp.raise_for_status()
            print(f"fetched: {url}")
            raw = await resp.read()
            try:
                return raw.decode(resp.charset or "utf-8")
            except UnicodeDecodeError as e:
                for enc in ("utf-8-sig", "latin-1", "windows-1252"):
                    try:
                        return raw.decode(enc)
                    except UnicodeDecodeError:
                        pass
                return e
    except aiohttp.ClientResponseError as e:
        print(f"[ERROR] on fetch: {e}")
        return e
    except asyncio.TimeoutError as e:
        print(f"[ERROR] timeout: {url}")
        return e
    except Exception as e:
        print(f"[ERROR] generic on fetch: {str(e)}")

## writer task
async def writer_task(output_path: str, file_name: str, queue: asyncio.Queue, write: bool = True):
    while not write:
        item = await queue.get()
        if item == None: break
        continue

    os.makedirs(output_path, exist_ok=True)
    async with aiofiles.open(os.path.join(output_path, file_name), "a") as file:
        while write:
            item = await queue.get()
            if item == None: break

            line = json.dumps(item)
            await file.write(line+"\n")
            queue.task_done()

def to_arrow(value):
    if value is None: return None

    ## base types
    if isinstance(value, (str, int, float, bool)): return value

    ## recursive call on list
    if isinstance(value, list): return [to_arrow(v) for v in value]

    ## recursive call on dict
    if isinstance(value, dict): return {k: to_arrow(v) for k, v in value.items()}

    ## fallback to string conversion
    return str(value)

def normalize_row(row: dict):
    return {k: to_arrow(v) for k, v in row.items()}

## writer task for parquet file format
async def writer_task_parquet(output_path: str, file_name: str, queue: asyncio.Queue, flush_every: int = 100):
    ## set up task
    os.makedirs(output_path, exist_ok=True)
    parquet_path = os.path.join(output_path, file_name)
    tmp_path = parquet_path + ".tmp"
    buffer = []
    schema = None
    writer: pq.ParquetWriter| None = None

    def _write_chunk():
        nonlocal writer, buffer, schema
        if not buffer: return

        table = pa.Table.from_pylist(buffer, schema=schema)
        if writer == None: writer = pq.ParquetWriter(tmp_path, schema, use_dictionary=True, compression="SNAPPY")
        writer.write_table(table)
        buffer = []

    loop = asyncio.get_running_loop()

    try:
        while True:
            row = await queue.get()
            if row == None: break ## shutdown code

            buffer.append(normalize_row(row))
            if schema == None: schema = pa.Table.from_pylist([buffer[0]]).schema

            if len(buffer) >= flush_every:await loop.run_in_executor(None, _write_chunk)
    
        if buffer: await loop.run_in_executor(None, _write_chunk)

    finally:
        if writer: writer.close()
        if os.path.exists(tmp_path): os.replace(tmp_path, parquet_path)
        

## i/o & cpu controller function
async def fetch_process_write(article: dict, session: aiohttp.ClientSession, pool: ProcessPoolExecutor, writer_queue: asyncio.Queue, error_queue: asyncio.Queue, text_queue: asyncio.Queue, delay: float):
    ## fetches our raw html from the url using our session
    html = await fetch_html(url=article["url"], domain=article["media_url"], session=session, min_delay=delay)
    if isinstance(html, aiohttp.ClientResponseError): 
        await error_queue.put({**article, "error_type": "http", "error": {"status_code": html.status, "message": html.message}})
        return
    elif isinstance(html, UnicodeDecodeError):
        await error_queue.put({**article, "error_type": "decode", "error": str(html)})
        return
    elif isinstance(html, asyncio.TimeoutError): 
        await error_queue.put({**article, "error_type": "timeout", "error": str(html)})
        return
    elif isinstance(html, Exception):
        await error_queue.put({**article, "error_type": "connection", "error": str(html)})
        return
    elif not isinstance(html, str): 
        await error_queue.put({**article, "error_type": "http", "error": "html_none"})
        return
    
    try:
        ## grabs our loop set up back in main()
        loop = asyncio.get_running_loop()
        ## runs our synchronous function w/ our pool
        result = await loop.run_in_executor(pool, process_html, html, article["title"])
    except Exception as e:
        await error_queue.put({**article, "error_type": "processing", "error": str(e)})
        return

    if result.get("text"):
        just_text = {**article, "text": result["text"]}
        await text_queue.put(just_text)

    if result.get("ok"): 
        del result["ok"]
        result = article | result
        await writer_queue.put(result)
        
    else: 
        if result.get("ok") != None: del result["ok"]
        if result.get("text") != None: del result["text"]
        result = article | result
        await error_queue.put(result)

domain_limits = {}
def get_domain_sem(media_url: str, lim: int) -> asyncio.Semaphore:
    if media_url not in domain_limits: domain_limits[media_url] = asyncio.Semaphore(lim)
    return domain_limits[media_url]

async def main_async(args):
    ## grab data using pandas
    data = read_csv(args.input)[["id", "media_url", "url", "title", "publish_date"]]
    articles = data.to_dict("records")

    writer_queue = asyncio.Queue() ## refresh writer queue
    writer = asyncio.create_task(writer_task_parquet(args.output, "output.parquet", writer_queue, args.flush)) ## set up task

    text_queue = asyncio.Queue()
    text_writer = asyncio.create_task(writer_task(args.output, "cleaned_texts.jsonl", text_queue, args.text))

    error_queue = asyncio.Queue()
    error_writer = asyncio.create_task(writer_task(args.output, "errors.jsonl", error_queue))

    ## create our worker pool (this will manage out cpu bound processing!!)
    with ProcessPoolExecutor(max_workers=args.cores, initializer=init_worker) as pool:
        ## create out http session which will manage our i/o bound getting
        async with aiohttp.ClientSession() as session:
            http_sem = asyncio.Semaphore(args.con_gets) ## num of concurrent url gets

            ## our semaphore controlled async function
            async def semmedAsync(article): 
                dom_sem = get_domain_sem(article['media_url'], args.con_dom)
                ## pawns the semaphore controlled async to our session-pool controller
                async with dom_sem: 
                    async with http_sem: return await fetch_process_write(
                        article=article,
                        session=session, 
                        pool=pool, 
                        writer_queue=writer_queue, 
                        error_queue=error_queue,
                        text_queue=text_queue,
                        delay=args.min_del
                    )
            
            ## making running loop (this is what asyncio returns with get_running_loop())
            tasks = [semmedAsync(article) for article in articles]
            try:
                await asyncio.gather(*tasks)
            finally:
                ## Puts in a final shutdown signal for the writers (None) and waits for the task to fully finish
                await asyncio.gather(writer_queue.put(None), error_queue.put(None), text_queue.put(None))
                await asyncio.gather(asyncio.shield(writer), error_writer, text_writer)
    print(f"[DONE]")

def main(args):
    if platform.system() == "Linux": preload_models()
    asyncio.run(main_async(args))
