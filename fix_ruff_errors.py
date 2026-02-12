import re

def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Fix SIM108 in embedding.py
    # Pattern:
    #             if isinstance(request.input, str):
    #                 inputs = [request.input]
    #             else:
    #                 inputs = request.input

    pattern = r'if isinstance\(request\.input, str\):\s+inputs = \[request\.input\]\s+else:\s+inputs = request\.input'
    replacement = 'inputs = [request.input] if isinstance(request.input, str) else request.input'

    # We need to handle indentation.
    # The pattern above doesn't account for indentation.
    # Let's match indentation.

    lines = content.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'if isinstance(request.input, str):' in line:
            indent = line[:line.find('if')]
            if i+3 < len(lines) and                'inputs = [request.input]' in lines[i+1] and                'else:' in lines[i+2] and                'inputs = request.input' in lines[i+3]:
                new_lines.append(f'{indent}inputs = [request.input] if isinstance(request.input, str) else request.input')
                i += 4
                continue
        new_lines.append(line)
        i += 1

    with open(filepath, "w") as f:
        f.write('\n'.join(new_lines))

fix_file("src/ai_workers/workers/embedding.py")
