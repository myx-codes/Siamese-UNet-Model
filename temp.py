from modules.counting import counting

if __name__ == "__main__":
    working_dir = "./data/working"
    predicted_dir = "./outputs/predicted"
    output_csv = "./outputs/predicted/counting.csv"
    counting(working_dir, predicted_dir, output_csv)