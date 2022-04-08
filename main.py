from models.model import main
from os import path

if __name__ == '__main__':
    if not path.exists('./models/final_model.pickle'):
        main(True)
    else:
        main(False)
    exit(0)
