from langchain_core.tools import tool
import sympy

@tool
def calculate(expression: str) -> str:
    """Calculates a mathematical expression. Use for math questions."""
    try:
        # Evaluate safely using sympy
        result = sympy.sympify(expression).evalf()
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"