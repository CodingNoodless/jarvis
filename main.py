from core.listen import listen
from core.speak import speak
from core.parser import parse
from core.dispatcher import dispatch
from core.context import context

USE_VOICE = True

def main():
    print("Jarvis is running. Say 'exit' to quit.")

    while True:
        if USE_VOICE:
            text = listen()
        else:
            text = input("You: ")

        if "exit" in text.lower():
            break

        intent = parse(text)
        response = dispatch(intent, context=context)
        speak(response)

if __name__ == "__main__":
    main()
