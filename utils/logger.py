import os

class Logger:
    def __init__(self, file_path, verbose=True):
        print()

        if not os.path.exists("/".join(file_path.split("/")[:-1])):
            os.mkdir("/".join(file_path.split("/")[:-1]))
            
        self.file_path = file_path
        self.verbose = verbose

    def __call__(self, message, verbose=None):
        with open(self.file_path, "a") as f:
            f.write(f"{message}\n")

            if verbose:
                print(message)
            elif self.verbose and verbose is not None:
                print(message)



if __name__ == "__main__":
    logger = Logger("./logs/log.txt") 
    logger("Hello, World!")