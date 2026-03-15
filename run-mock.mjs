import http from "http";

const port = 5555;

const server = http.createServer((req, res) => {
	if (req.method === "POST" && req.url === "/v1/chat/completions") {
		let body = "";

		req.on("data", (chunk) => {
			body += chunk;
		});

		req.on("end", () => {
			const response = {
				id: "mock-123",
				object: "chat.completion",
				created: Date.now(),
				model: "mock-model",
				choices: [
					{
						index: 0,
						message: {
							role: "assistant",
							content: `
factorial.cpp
\`\`\`cpp
#include <iostream>
using namespace std;

long factorial(int n){
    if(n<=1) return 1;
    return n * factorial(n-1);
}

int main(){
    int n = 5;
    cout << "Factorial of " << n << " = " << factorial(n) << endl;
}
\`\`\`

prime.cpp
\`\`\`cpp
#include <iostream>
using namespace std;

bool isPrime(int n){
    if(n<2) return false;

    for(int i=2;i*i<=n;i++){
        if(n%i==0) return false;
    }

    return true;
}

int main(){
    for(int i=1;i<=50;i++){
        if(isPrime(i)){
            cout << i << " ";
        }
    }
}
\`\`\`

factorial.py
\`\`\`python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

print("Factorial:", factorial(5))
\`\`\`

prime.py
\`\`\`python
def is_prime(n):
    if n < 2:
        return False

    for i in range(2,int(n**0.5)+1):
        if n % i == 0:
            return False

    return True

for i in range(1,50):
    if is_prime(i):
        print(i,end=" ")
\`\`\`

Factorial.java
\`\`\`java
public class Factorial {

    static int factorial(int n){
        if(n<=1) return 1;
        return n * factorial(n-1);
    }

    public static void main(String[] args){
        System.out.println("Factorial = " + factorial(6));
    }
}
\`\`\`

Prime.java
\`\`\`java
public class Prime {

    static boolean isPrime(int n){
        if(n<2) return false;

        for(int i=2;i*i<=n;i++){
            if(n%i==0) return false;
        }

        return true;
    }

    public static void main(String[] args){
        for(int i=1;i<=50;i++){
            if(isPrime(i)){
                System.out.print(i + " ");
            }
        }
    }
}
\`\`\`

frontend/index.html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Todo App</title>
<link rel="stylesheet" href="style.css">
</head>
<body>

<h1>Todo List</h1>

<input id="taskInput" placeholder="Enter task">
<button onclick="addTask()">Add</button>

<ul id="taskList"></ul>

<script src="app.js"></script>

</body>
</html>

frontend/style.css
\`\`\`css
body{
font-family:Arial;
background:#f2f2f2;
text-align:center;
}

ul{
list-style:none;
padding:0;
}

li{
background:white;
margin:10px;
padding:10px;
border-radius:5px;
}
\`\`\`

frontend/app.js
\`\`\`javascript
function addTask(){

const input = document.getElementById("taskInput")
const list = document.getElementById("taskList")

if(input.value.trim() === "") return

const li = document.createElement("li")
li.textContent = input.value

list.appendChild(li)

input.value = ""
}
\`\`\`

public/server.js
\`\`\`javascript
const express = require("express")
const sqlite3 = require("sqlite3").verbose()

const app = express()
const db = new sqlite3.Database("movies.db")

app.use(express.json())
app.use(express.static("public"))

db.run(\`
CREATE TABLE IF NOT EXISTS movies(
id INTEGER PRIMARY KEY AUTOINCREMENT,
title TEXT,
director TEXT,
genre TEXT,
rating INTEGER
)
\`)

app.get("/movies",(req,res)=>{
db.all("SELECT * FROM movies",(err,rows)=>{
res.json(rows)
})
})

app.listen(3000,()=>console.log("Server running on 3000"))
\`\`\` 
			 
			  `,
						},
						finish_reason: "stop",
					},
				],
				usage: {
					prompt_tokens: 10,
					completion_tokens: 20,
					total_tokens: 30,
				},
			};

			res.writeHead(200, {
				"Content-Type": "application/json",
			});

			res.end(JSON.stringify(response));
		});
	} else {
		res.writeHead(404);
		res.end();
	}
});

server.listen(port, () => {
	console.log("Mock OpenAI server running on port", port);
});
