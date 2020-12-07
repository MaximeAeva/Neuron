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

More about sources [Here](https://github.com/MaximeAeva/Neuron/blob/master/src/README.md)

More about Neural style transfert on tensorflow [here](https://github.com/MaximeAeva/NST)

## About
When training, CNN display information regarding its work
```console
----------Forward prop----------
A1: (64, 28, 28, 6)
A2: (64, 14, 14, 6)
A3: (64, 10, 10, 16)
A4: (64, 5, 5, 16)
A5: (400, 64)
A6: (120, 64)
A7: (84, 64)
A8: (10, 64)
----------Cost computation----------
Cost: 1.622069
----------Bacward prop----------
dA9: (10, 64)
dA8: (84, 64)
dA7: (120, 64)
dA6: (400, 64)
dA5: (64, 5, 5, 16)
dA4: (64, 10, 10, 16)
dA3: (64, 14, 14, 6)
dA2: (64, 28, 28, 6)
----------Time spent----------
forward conv: 65.423750
forward pool: 7.893981
forward dense: 0.007658
backward conv: 101.860744
backward pool: 17.962010
backward dense: 0.007316
```

CNN architecture can be parse following the schema thereafter
```console
AlexNet = (('input', (224, 224, 3)),
           ('conv', (8, 3, 96, 0, 4)),('pool', (3, 2), 'max'), 
           ('conv', (5, 96, 256, 2, 0)), ('pool', (3, 2), 'max'), 
           ('conv', (3, 256, 384, 1, 0)), ('conv', (3, 384, 384, 1, 0)), 
           ('conv', (3, 384, 256, 1, 0)), ('pool', (3, 2)), 
           ('flatten', 9216), 
           ('dense', 4096, 'relu'), ('dense', 4096, 'relu'),
           ('dense', 1000, 'relu'), 
           ('dense', 10, 'sigmoid'))
```
Error will occur if you do not parse layers in a way that ouput layers l-1 shape fit input l layer shape

## Installation
Windows:

```console
git clone
```
Then run train.py will train different network architectures until it finds one below your specified accuracy.

## Usage example

Split data into classes

## Release History

*0.2.0
|   Running CNN -> Try Dask and UCX + remove unnecessary files
*0.1.0
|   Almost done CNN (some bug fix in bacward)
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
