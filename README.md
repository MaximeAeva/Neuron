# Neuron
> A CUDA boosted CNN implementation from scratch

I used weather data (temperature, pression, rain, wind ...) to build and test my neural network.

Thereafter a gif pointing out the training session (each image is separated from the previous by 1000 epochs)

![Hey!](https://github.com/MaximeAeva/Neuron/blob/master/res/hello.gif)

The result (orange) on temperature data (blue)

![Result!](https://github.com/MaximeAeva/Neuron/blob/master/res/solution.png)

This neural network can predict temperature within an interval below +- 2°C.

This neural network points out that it is really difficult to predict wind direction as it is almost a random function.
The same thing appear for rain since levels are almost bolean and my output function is a sigmoid.

Despite these problems, this works really well on temperature and pression.

My own thought is, weather forecast will always be better forecast from model (as it rely on differential equation).

## Installation
Windows:

```console
git clone
```
Then run train.py will train different network architectures until it finds one below your specified accuracy.
CUDA neural network is not complete for now (in fact, neural network is, but CNN isn't).
Tests tend to forecast a x30 computational speed improvement !

## Usage example

Split data into classes

## Release History

*0.0.0
|   Rebuild the github (started the project last year where github was out of my scope)

## Meta

Me – [MaximeAeva](https://github.com/MaximeAeva) – Feel free to contact

## Contributing

1. Fork it (<https://github.com/MaximeAeva/Neuron/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
