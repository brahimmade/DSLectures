from os import RTLD_NOW
import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.cache['x']=x
        l1 = torch.matmul(x,torch.t(self.parameters['W1'])) + self.parameters['b1']
        self.cache['l1'] = l1
        if self.f_function == "relu":
            m = torch.nn.ReLU()
            f = m(l1)
        elif self.f_function == "sigmoid":
            m = torch.nn.Sigmoid()
            f = m(l1)
        elif self.f_function == "identity":
            f = l1
        self.cache['f'] = f
        l2 = torch.matmul(f,torch.t(self.parameters['W2'])) + self.parameters['b2']
        self.cache['l2'] = l2
        if self.g_function == "relu":
            m = torch.nn.ReLU()
            g = m(l2)
        elif self.g_function == "sigmoid":
            m = torch.nn.Sigmoid()
            g = m(l2)
        elif self.g_function == "identity":
            g = l2
        self.cache['g'] = g
        return g
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """ 
        def relu_grad(inp):
            # grad of relu with respect to input activations
            return (inp>0).float()
        dfdl1 = relu_grad(self.cache['l1'])

        dgdl2 = self.parameters['W2']
        dl1dw1 = self.cache['x']
        dl2df = self.cache['f']

        def mse_grad(inp, targ): 
            # grad of loss with respect to output of previous layer
            inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]

        def relu_grad(inp, out):
            # grad of relu with respect to input activations
            inp.g = (inp>0).float() * out.g

        def lin_grad(inp, out, w, b):
            # grad of matmul with respect to input
            inp.g = out.g @ w.t()
            w.g = inp.t() @ out.g
            b.g = out.g.sum(0)

        def forward_and_backward(inp, targ):
            # forward pass:
            l1 = inp @ w1 + b1
            
            out = l2 @ w2 + b2
            l2 = relu(l1)
            
            # backward pass:
            mse_grad(out, targ)
            lin_grad(l2, out, w2, b2)
            relu_grad(l1, l2)
            lin_grad(inp, l1, w1, b1)


        self.grads['dJdW2'] = torch.matmul(torch.t(dJdy_hat),dl2df)
        self.grads['dJdb2'] = torch.sum(dJdy_hat,axis=0)
        self.grads['dJdb1'] = torch.sum(torch.matmul(torch.t(dfdl1),dJdy_hat @ dgdl2),axis=0)
        self.grads['dJdW1'] = torch.matmul(torch.matmul((torch.matmul(dJdy_hat,dgdl2)),torch.t(dfdl1)),dl1dw1)
        
       

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    diff = y_hat - y
    loss = torch.mean(torch.square(diff))
    dJdy_hat = torch.mul(2.0, diff) / (diff.shape[0] * diff.shape[1])

    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    pass

    # return loss, dJdy_hat
