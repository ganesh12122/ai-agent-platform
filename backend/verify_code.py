import asyncio
import sys
import os

# Ensure backend directory is in path
sys.path.append(os.getcwd())

from tools import execute_tool

async def test():
    print("Testing Code Executor Tool...")
    code = """
def prime_check(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(20) if prime_check(n)]
print(f'Primes under 20: {primes}')
"""
    try:
        result = await execute_tool('code_executor', code=code)
        print(f"Result: {result}")
        
        expected_substr = "[2, 3, 5, 7, 11, 13, 17, 19]"
        if expected_substr in str(result.get('result', '')):
            print("Success ✅")
        else:
            print("Output Mismatch ❌")
            
    except Exception as e:
        print(f"Error ❌: {e}")

if __name__ == "__main__":
    asyncio.run(test())
