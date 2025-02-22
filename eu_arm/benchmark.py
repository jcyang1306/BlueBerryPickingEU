import time
from functools import wraps

def benchmark(func):
    """
    Decorator to benchmark the time cost of a function.
    """
    @wraps(func)  # Preserves the original function's metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"  >> benchmark >> Function <{func.__name__}> done, [{elapsed_time:.4f}]s used")
        return result  # Return the result of the original function
    return wrapper

# Example usage
@benchmark
def example_function(n):
    """Example function that sleeps for `n` seconds."""
    time.sleep(n)
    return "Done"

# Test the decorated function
if __name__ == "__main__":
    example_function(2)  # Call the function with a 2-second sleep