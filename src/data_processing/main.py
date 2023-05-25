import sys
from data_collection.script import main_collector
from data_cleaning.script.script import main_cleaner
from tokenizing.tokenizer import apply_main_tokenizer


if __name__ == "__main__":
    args = sys.argv

    if len(args) == 1:
        print()
        print("the correct arguemnts arguments are `collect`, `clean`, `tokenize`, `all`")
        print("run command is `python main.py arg`")
        print()
    else:
        second_arg = args[1]

        if second_arg == "collect":
            main_collector()

        elif second_arg == "clean":
            main_cleaner()

        elif second_arg == "tokenize":
            apply_main_tokenizer()

        elif second_arg == "all":
            main_collector()
            main_cleaner()
            apply_main_tokenizer()

        else:
            print()
            print("the correct arguemnts arguments are `collect`, `clean`, `tokenize`, `all`")
            print("running command is `python main.py arg`")
            print()
