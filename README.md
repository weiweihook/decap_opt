# decap_opt

## Prerequisites
We model the 2.5D hierarchical PDN  by using 'bin' directory, there are three main files that we will use, as follows:
 -[diegen]: generates the on-chip PDN according to the chip PDN configuration file (eg. config/case/chiplet1.conf)
 -[intgen]: generates the overall PDN according to the overall PDN configuration file (eg. config/case/intp_chip1.conf)
 -[inttrvmap]: generates VVI diatribution according to the time-domain VVI data 

The circuit simulator NGSPICE is then  employed for impedance and VVI analysis. You must install it.
```shell 
pip install ngspice
```



## Experiment Settings 
<center>
  
| Hyperparameter | Value |
| :-------------------------:|:-------------------------: |
| Activation Function | Linear rectification function|
| Optimizer           | Adam |
| Learning Rate       | 2.5E-4 |
| Clip Gradient Norm  | 0.5 |
| Total Epoch         | 600 |
| Batch Size          | 480 |
| Minibatch Size      | 4   |
| Clipping Coefficient| 0.1 |
| Entropy Coefficient | 0.01|
| Value Coefficient   | 0.5 |
| Discount(γ)         | 0.99|

</center>

```shell 
pip install ngspice
```