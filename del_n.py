import sys

def remove_line_breaks(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        content_one_line = content.replace('\n', '').replace('\r', '')
        
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content_one_line)
        
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("python3 remove_line_breaks.py text.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    remove_line_breaks(filename)

if __name__ == "__main__":
    main()
