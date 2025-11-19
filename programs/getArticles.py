import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor
from pandas import read_csv

## cpu bound synchronous processing.
def process_html(html: str):
    print(html)

## simple html fetch
async def fetch_html(url: str, session: aiohttp.ClientSession):
    try:
        async with session.get(url, timeout=10.0) as resp:
            resp.raise_for_status()
            return await resp.text()
    except Exception as e:
        print(e)

## i/o & cpu controller function
async def fetch_process_write(url, session: aiohttp.ClientSession, pool: ProcessPoolExecutor, outFile):
    ## fetches our raw html from the url using our session
    html = await fetch_html(url=url, session=session)
    if html == None: return None

    ## grabs our loop set up back in main()
    loop = asyncio.get_running_loop()
    ## runs our synchronous function w/ our pool
    future = loop.run_in_executor(pool, process_html, html)
    ## await the results
    result = await future

    ## todo: write to file (filetype??)

async def main(args, callback=None):
    ## grab data using pandas
    data = read_csv(args.input)[["url", "media_url"]]
    urls = data["url"]

    ## create our worker pool (this will manage out cpu bound processing!!)
    with ProcessPoolExecutor(max_workers=args.cores) as pool:
        ## create out http session which will manage our i/o bound getting
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(args.con_gets) ## num of concurrent url gets

            ## our semaphore controlled async function
            async def semmedAsync(url): 
                ## pawns the semaphore controlled async to our session-pool controller
                async with sem: return await fetch_process_write(url, session, pool, args.output)
            
            ## making running loop (this is what asyncio returns with get_running_loop())
            tasks = [semmedAsync(url) for url in urls]
            futures = await asyncio.gather(*tasks)