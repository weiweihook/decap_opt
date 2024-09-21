# decap_opt

## Prerequisites
We model the 2.5D hierarchical PDN  by using 'bin' directory, there are three main files that we will use, as follows:
 - **diegen**: generates the on-chip PDN and subckts based on the chip PDN configuration (e.g.  config/case/chiplet1.conf), the usage is as follows:
```shell 
$ bin/diegen chiplet1.conf
```
 - **intgen**: generates the overall PDN and simulation file (.sp file) for the 2.5D architecture, incorporating both the on-chip and interposer PDNs, according to the PDN configuration (e.g.  config/case/intp_chip1.conf), the usage is as follows:
```shell 
$ bin/intgen intp_chip1.conf
```
 - **inttrvmap**: generates the Voltage Violation Index (VVI) distribution across the on-chip PDN based on time-domain VVI data , the usage is as follows:
```shell 
$ bin/inttrvmap intp_chip1.conf vdi.raw 'Vdd' 'Ripple'
```
There, 'Vdd' and 'Ripple' need to be replaced by the specifics. 'vdi.raw' is generated by NGSPICE, will be demonstrated later.

The circuit simulator NGSPICE is then  employed for impedance and VVI analysis. You must install it.
```shell 
$ pip install ngspice
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