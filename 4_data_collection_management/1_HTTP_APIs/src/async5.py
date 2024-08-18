import asyncio

async def main():
    task = asyncio.create_task(bye())
    print("Hello!")

async def bye():
    print("OK")
    await asyncio.sleep(1) # forces the program to wait for one second
    print("Goodbye!")  
asyncio.run(main())
