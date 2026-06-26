def greet():
    return "hello from greet()"


print("this top level line always runs.")

if __name__ == "__main__":
    print("i am being run directly!")
    print(greet())
