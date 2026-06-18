import preprocess
import train_supcon
import train_openmax
import evaluate

if __name__ == "__main__":
    print("Running preprocessing...")
    preprocess.main()

    print("Training SupCon model...")
    train_supcon.main()

    print("Training OpenMax model...")
    train_openmax.main()

    print("Evaluating model...")
    evaluate.main()
