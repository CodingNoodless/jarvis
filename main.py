from core.parser import parse
from core.dispatcher import dispatch

def main():
    print("Jarvis CLI Assistant. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        intent_obj = parse(user_input)
        response = dispatch(intent_obj)
        print(f"Jarvis: {response}")

if __name__ == "__main__":
    main()