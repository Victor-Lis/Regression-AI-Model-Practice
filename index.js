// Esse é o arquivo que usei para construir as tabelas de dados.

const f1 = (x) => console.log(`${x},${x*2+1}`)

console.log("Function 1")
console.log("x,y")

for(let x = 1; x <= 10; x++){
    f1(x)
}

console.log("")

const f2 = (x) => console.log(`${x},${x*4+1}`)

console.log("Function 2")
console.log("x,y")

for(let x = 1; x <= 100; x++){
    f2(x)
}

console.log("")

const f3 = (x) => console.log(`${x},${(x*3)+(x/2)}`)

console.log("Function 3")
console.log("x,y")

for(let x = 1; x <= 1000; x++){
    f3(x)
}

console.log("")
console.log("User Testing")
f3(3201)
console.log("")