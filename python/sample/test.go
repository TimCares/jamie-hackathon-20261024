package main

import (
	"encoding/json"
	"flag"
	"fmt"
)

func factorial(n int) int {
	result := 1
	for i := 2; i <= n; i++ {
		result *= i
	}
	return result
}
type FactorialInput struct {
	Number int `json:"number"`
}

func factorialInputFromJSON(data []byte) (FactorialInput, error) {
	var f FactorialInput
	err := json.Unmarshal(data, &f)
	return f, err
}

func main() {
	var input FactorialInput
	flag.IntVar(&input.Number, "n", 0, "number to compute the factorial of")
	flag.Parse()

	fmt.Println(factorial(input.Number))
}
