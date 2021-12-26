import awkward as ak

array = ak.Array([
    ["lst", [1, 2, 3, 4]],
    {"msg": "message text"},
    {"nested": {
        "array": [10, 20, 30]
    }}
])

print("Awkward Array:", array)
print("Type:", ak.type(array))

print(array[2]["nested", "array", 1])
