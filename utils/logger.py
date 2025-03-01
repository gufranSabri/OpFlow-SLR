import os
from tqdm import tqdm

class Logger:
    def __init__(self, file_path):
        print()

        if not os.path.exists("/".join(file_path.split("/")[:-1])):
            os.mkdir("/".join(file_path.split("/")[:-1]))
            
        self.file_path = file_path

    def __call__(self, message, verbose=True):
        with open(self.file_path, "a") as f:
            f.write(f"{str(message)}\n")

            if verbose:
                tqdm.write(str(message))
                # print(message)



if __name__ == "__main__":
    logger = Logger("./logs/log.txt") 
    logger("Hello, World!")