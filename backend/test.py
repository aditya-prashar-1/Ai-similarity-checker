import requests

url = "http://127.0.0.1:5000/compare"
data = {
    "problem": "Print all elements in a linked list.",
    "user_code": "#include <iostream>\nusing namespace std;\nint main() { cout << \"Hello\"; }"
}

res = requests.post(url, json=data)
print(res.json())


