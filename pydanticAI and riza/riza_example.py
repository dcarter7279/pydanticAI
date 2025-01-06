import rizaio

def execute_code(code: str) -> str:
    print(f"Executing code...", code)
    riza = rizaio.Riza()
    result = riza.command.exec(language="PYTHON", code=code)
    return result.stdout

if __name__ == "__main__":
    print(execute_code("print('Hello, World!')"))
    
    