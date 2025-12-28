print("[DEBUG __main__.py] Starting __main__.py execution")
from .cli import main
print("[DEBUG __main__.py] Imported main from cli")
if __name__ == "__main__":
    print("[DEBUG __main__.py] Calling main()")
    main()
    print("[DEBUG __main__.py] main() returned")
