# Backpropagation for a Linear Layer 

Justin Johnson 

April 19, 2017 

In these notes we will explicitly derive the equations to use when backpropagating through a linear layer, using minibatches. 

During the forward pass, the linear layer takes an input _X_ of shape _N × D_ and a weight matrix _W_ of shape _D × M_ , and computes an output _Y_ = _XW_ of shape _N × M_ by computing the matrix product of the two inputs. To make things even more concrete, we will consider the case _N_ = 2, _D_ = 2, _M_ = 3. We can then write out the forward pass in terms of the elements of the inputs: 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0001-05.png)



![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0001-06.png)



![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0001-07.png)


After the forward pass, we assume that the output will be used in other parts of the network, and will eventually be used to compute a scalar loss _L_ . During the backward pass through the linear layer, we assume that the derivative _∂Y∂L_<sup>hasalreadybeencomputed.Forexampleifthelinearlayeris</sup> part of a linear classifier, then the matrix _Y_ gives class scores; these scores are fed to a loss function (such as the softmax or multiclass SVM loss) which computes the scalar loss _L_ and derivative _∂Y∂L_<sup>ofthelosswithrespecttothe</sup> scores. 

Since _L_ is a scalar and _Y_ is a matrix of shape _N × M_ , the gradient _∂Y∂L_ will be a matrix with the same shape as _Y_ , where each element of _∂Y_<sup>_<u>∂L</u>_givesthe</sup> derivative of the loss _L_ with respect to one element of _Y_ : 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0001-10.png)


During the backward pass our goal is to use _∂Y_<sup>_<u>∂L</u>_in order to compute</sup> _∂X_<sup>_<u>∂L</u>_and</sup> _∂W∂L_<sup>.Again,since</sup><sup>_L_isascalarweknowthat</sup> _∂X_<sup>_<u>∂L</u>_musthavethesameshapeas</sup> _X_ ( _N × D_ ) and _∂W∂L_<sup>musthavethesameshapeas</sup><sup>_W_(</sup><sup>_D × M_).</sup> 

1 

By the chain rule, we know that: 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0002-01.png)


The terms _∂X_<sup>_<u>∂Y</u>_and</sup> _∂W∂Y_<sup>inEquation5are</sup><sup>_Jacobianmatrices_containingthe</sup> partial derivative of each element of _Y_ with respect to each element of the inputs _X_ and _W_ . However we do not want to form the Jacobian matrices _∂X_<sup>_<u>∂Y</u>_and</sup> _∂W_<sup>_<u>∂Y</u>_explicitly,</sup> because they will be very large. In a typical neural network we might have _N_ = 64 and _M_ = _D_ = 4096; then _∂X_<sup>_<u>∂Y</u>_consists of 64</sup><sup>_·_4096</sup><sup>_·_64</sup><sup>_·_4096 scalar values;</sup> this is more than 68 billion numbers; using 32-bit floating point, this Jacobian matrix will take 256 GB of memory to store. Therefore it is completely hopeless to try and explicitly store and manipulate the Jacobian matrix. However it turns out that for most common neural network layers, we can derive expressions that compute the product _∂X_<sup>_<u>∂Y</u>_</sup> _∂Y∂L_<sup>_withoutexplicitlyforming_</sup> _the Jacobian ∂X∂Y_<sup>.Evenbetter,wecantypicallyderivethisexpressionwithout</sup> even computing an explicit expression for the Jacobian _∂X_<sup>_<u>∂Y</u>_;inmanycaseswe</sup> can work out a small case on paper and then infer the general formula. Let’s see how this works out for our specific case of _N_ = 2, _D_ = 2, _M_ = 3. We first tackle _∂X_<sup>_<u>∂L</u>_.Again,weknowthat</sup> _∂X_<sup>_<u>∂L</u>_musthavethesameshapeas</sup><sup>_X_:</sup> 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0002-03.png)


We can proceed one element of a time. First we will compute _∂x∂L_ 1 _,_ 1<sup>.Bythe</sup> chain rule, we know that 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0002-05.png)


In the above equation _L_ and _x_ 1 _,_ 1 are scalars so _∂x∂L_ 1 _,_ 1<sup>isalsoascalar.Ifwe</sup> view _Y_ not as a matrix but as a collection of intermediate scalar variables, then we can use the chain rule to write _∂x∂L_ 1 _,_ 1<sup>solelyintermsofscalarderivatives.</sup> 

To avoid working with sums, it is convenient to collect all terms _∂y∂Li,j_<sup>into</sup> a single matrix _∂Y_<sup>_<u>∂L</u>_;here</sup><sup>_L_isascalarand</sup><sup>_Y_isamatrix,so</sup> _∂Y_<sup>_<u>∂L</u>_hasthesame</sup> shape as _Y_ ( _N × M_ ), where each element of _∂Y_<sup>_<u>∂L</u>_givesthederivativeof</sup><sup>_L_with</sup> _∂yi,j_ respect to one element of _Y_ . We similarly collect all terms _∂x_ 1 _,_ 1<sup>intoasingle</sup> matrix _∂x∂Y_ 1 _,_ 1<sup>;since</sup><sup>_Y_isamatrixand</sup><sup>_x_1</sup><sup>_,_1isascalar,</sup> _∂x∂Y_ 1 _,_ 1<sup>isamatrixwiththe</sup> same shape as _Y_ ( _N × M_ ). Since _∂x∂L_ 1 _,_ 1<sup>isascalar,weknowthattheproductof</sup> _∂Y_<sup>_<u>∂L</u>_and</sup> _∂x∂Y_ 1 _,_ 1<sup>mustbea</sup> scalar; by inspecting the expression using only scalar derivatives, it is clear that in this context the product of _∂Y_<sup>_<u>∂L</u>_and</sup> _∂x∂Y_ 1 _,_ 1<sup>mustbeadotproduct.</sup> 

2 

In the backward pass we are already given _∂Y_<sup>_<u>∂L</u>_,soweonlyneedtocompute</sup> _∂x∂L_ 1 _,_ 1<sup>;wecaneasilycomputethisfromEquation3:</sup> 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0003-01.png)


Now combining Equations 6, 7, and 8 gives: 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0003-03.png)


We can now repeat the process to compute the other entries of _∂X∂L_<sup>,one</sup> element at a time: 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0003-05.png)



![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0003-06.png)


Finally we can combine Equations 9, 14, 17, and 20 to give a single expression for _∂X_<sup>_<u>∂L</u>_intermsof</sup><sup>_W_and</sup> _∂Y_<sup>_<u>∂L</u>_:</sup> 

3 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0004-00.png)


In Equation 24, recall that _∂Y_<sup>_<u>∂L</u>_is a matrix of shape</sup><sup>_N ×M_and</sup><sup>_W_is a matrix</sup> of shape _D × M_ ; thus _∂X_<sup>_<u>∂L</u>_=</sup> _∂Y_<sup>_<u>∂L</u>W T_hasshape</sup><sup>_N × D_,whichisthesameshape</sup> as _X_ . 

We derived Equation 24 in the specific case of _N_ = _D_ = 2, _M_ = 3, but it holds for any values of _N_ , _D_ , and _M_ . This equation allows us to efficiently compute _∂X_<sup>_<u>∂L</u>_using</sup> _∂Y_<sup>_<u>∂L</u>_and</sup><sup>_W_,withoutexplicitlyformingtheJacobian</sup> _∂X_<sup>_<u>∂Y</u>_.</sup> Using the same strategy of thinking about components one at a time, you can derive a similarly simple equation to compute _∂W_<sup>_<u>∂L</u>_without explicitly forming</sup> the Jacobian _∂W∂Y_<sup>:</sup> 


![](references/papers/ml-foundations/linear-backprop_images/linear-backprop.pdf-0004-03.png)


In this equation _∂W∂L_<sup>musthavethesameshapeas</sup><sup>_W_(</sup><sup>_D × M_);ontheright</sup> hand side _X_ is a matrix of shape _N × D_ and _∂Y_<sup>_<u>∂L</u>_is a matrix of shape</sup><sup>_N ×M_,so</sup> the matrix-matrix product on the right will produce a matrix of shape _D × M_ . 

This strategy of thinking one element at a time can help you to derive equations for backpropagation for a layer even when the inputs and outputs to the layer are tensors of arbitrary shape; this can be particularly valuable for example when deriving backpropagation for a convolutional layer. 

4 

