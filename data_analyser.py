#Imports
import queue
import time
import threading
from pre_processing import PreProcessing
from classifier import Classifier

class DataInput:
    def __init__(self, data, timestamp) -> None:
        self.data = data
        self.timestamp = timestamp

    def get_time(self) -> str:
        return time.strftime("%d/%m/%Y, %H:%M:%S",self.timestamp)

class DataManagement:
    def __init__(self, capture_window) -> None:
        self.active = True
        self.capture_window = capture_window
        self.q = queue.Queue()
        self.proc = PreProcessing()
        self.cl = Classifier()
        self.thread = threading.Thread(target=self.queue_manager)
        self.thread.start()
        

    def push(self, data_input) -> None:
        self.q.put(data_input)

    def close(self) -> None:
        self.active = False

    def queue_manager(self) -> None:
        while self.active:
            if self.q.empty():
                time.sleep(self.capture_window)
            else:
                item = self.q.get()
                data, timestamp = item.data, item.get_time()
                data = self.proc.process(data)
                score = self.cl.predict(data)
                if score < 0.5: ## Queda
                    print(f"Queda detectada ({timestamp}) | Score: {score}")
                self.q.task_done()
        ## Processo principal foi interrompido, limpe a pilha
        with self.q.mutex:
            self.q.queue.clear()