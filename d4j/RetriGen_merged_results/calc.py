results = []

for i in range(1, 11):
    with open(f"{i}.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

        for line in lines:
            results.append(line)

print(len(set(results)))
