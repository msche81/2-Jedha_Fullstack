import asyncio

async def main():
    task = asyncio.create_task(bye())
    print("Hello!")
    await asyncio.sleep(2)

async def bye():
    print("OK")
    await asyncio.sleep(1) # forces the program to wait for one second
    print("Goodbye!")  
asyncio.run(main())
