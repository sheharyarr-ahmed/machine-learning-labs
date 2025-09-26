# testing.py

# A simple Python test to verify PyCharm setup

def greet(name):
    return f"Hello, {name}! 🎉 PyCharm is working fine."

if __name__ == "__main__":
    # Test function
    message = greet("Developer")
    print(message)

    # Additional check: simple math
    result = 5 + 7
    print(f"5 + 7 = {result}")

    # Final confirmation
    print("✅ Everything seems to be working!")