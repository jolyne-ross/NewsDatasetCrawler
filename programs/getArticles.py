#!/usr/bin/env python3
import asyncio
import aiohttp
import aiofiles
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pandas import read_csv

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

## cpu bound synchronous processing.
def process_html(html: str):
    print(html)

last_request_claim = {}
## simple html fetch w/ rate limiting
async def fetch_html(url: str, dom, session: aiohttp.ClientSession, min_delay):
    if min_delay>0:
        now = time.monotonic()
        last = last_request_claim.get(dom, 0)

        wait_time = last + min_delay - now
        if wait_time > 0: await asyncio.sleep(wait_time)

    last_request_claim[dom] = time.monotonic
    try:
        async with session.get(url, timeout=10.0, headers=HEADERS) as resp:
            resp.raise_for_status()
            print(f"fetched: {url}")
            return await resp.text()
    except Exception as e:
        print(f"[ERROR] on fetch: {e}")
        return e

## writer task
async def writer_task(output_path: str, queue: asyncio.Queue):
    if output_path == None: return
    async with aiofiles.open(output_path, "a") as file:
        while True:
            item = await queue.get()
            if item == None: break

            line = json.dumps(item)
            await file.write("\n"+line)
            queue.task_done()

## i/o & cpu controller function
async def fetch_process_write(url, dom, session: aiohttp.ClientSession, pool: ProcessPoolExecutor, writer_queue: asyncio.Queue, error_queue: asyncio.Queue, delay):
    ## fetches our raw html from the url using our session
    html = await fetch_html(url=url, dom=dom, session=session, min_delay=delay)
    if html == None: return None
    if isinstance(html, aiohttp.ClientResponseError): 
        await error_queue.put({"status_code"})
        return

    assert isinstance(html, str)

    ## grabs our loop set up back in main()
    loop = asyncio.get_running_loop()
    ## runs our synchronous function w/ our pool
    result = await loop.run_in_executor(pool, process_html, html)
    await writer_queue.put(result)
    
domain_limits = {}
def get_domain_sem(media_url, lim):
    if media_url not in domain_limits: domain_limits[media_url] = asyncio.Semaphore(lim)
    return domain_limits[media_url]

async def main_async(args):
    ## grab data using pandas
    data = read_csv(args.input)[["url", "media_url"]]
    urls = list(zip(data["url"], data["media_url"]))

    writer_queue = asyncio.Queue() ## refresh writer queue
    writer = asyncio.create_task(writer_task(args.output, writer_queue)) ## set up task

    error_queue = asyncio.Queue()
    error_writer = asyncio.create_task(writer_task(args.error_output, error_queue))

    ## create our worker pool (this will manage out cpu bound processing!!)
    with ProcessPoolExecutor(max_workers=args.cores) as pool:
        ## create out http session which will manage our i/o bound getting
        async with aiohttp.ClientSession() as session:
            http_sem = asyncio.Semaphore(args.con_gets) ## num of concurrent url gets

            ## our semaphore controlled async function
            async def semmedAsync(url, media_url): 
                dom_sem = get_domain_sem(media_url, args.con_dom)
                ## pawns the semaphore controlled async to our session-pool controller
                async with http_sem: 
                    async with dom_sem: return await fetch_process_write(
                        url=url, 
                        media_url=media_url, 
                        session=session, 
                        pool=pool, 
                        writer_queue=writer_queue, 
                        error_queue=error_queue,
                        delay=args.min_del
                    )
            
            ## making running loop (this is what asyncio returns with get_running_loop())
            tasks = [semmedAsync(url, media_url) for url, media_url in urls]
            await asyncio.gather(*tasks)

    ## Puts in a final shutdown signal for the writers (None) and waits for the task to fully finish
    await asyncio.gather([
        writer_queue.put(None),
        error_queue.put(None),
        writer,
        error_writer
    ])
    print(f"[DONE]")

def main(args, callback):
    asyncio.run(main_async(args))
