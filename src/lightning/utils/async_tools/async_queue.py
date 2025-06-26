import asyncio

class AsyncQueueManager:
    def __init__(self, threshold):
        self.queue = asyncio.Queue()
        self.threshold = threshold
        self.task_manager = None  # Don't start the task yet

    async def manage_tasks(self):
        """Monitor the queue and execute tasks when reaching the threshold."""
        try:
            while True:
                print(f"manager working... queue size: {self.queue.qsize()}")
                if self.queue.qsize() >= self.threshold:
                    tasks = []
                    while not self.queue.empty():
                        task = await self.queue.get()
                        tasks.append(task)
                    if tasks:
                        print(f"Executing {len(tasks)} tasks...")
                        await asyncio.gather(*tasks)
                await asyncio.sleep(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            

    def start_manager(self):
        """Start the task manager within an async context."""
        self.task_manager = asyncio.create_task(self.manage_tasks())
        print("Task manager started.")

    async def add_task(self, coro):
        """Add tasks to the queue."""
        await self.queue.put(coro)
        print(f"Task added to queue. Current queue size: {self.queue.qsize()}")
        print(coro)
        
    def is_empty(self):
        return self.queue.empty()