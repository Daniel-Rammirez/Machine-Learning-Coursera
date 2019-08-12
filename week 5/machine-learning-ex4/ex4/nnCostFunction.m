function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); %25x401
Theta2_grad = zeros(size(Theta2)); %10x26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Forward Propagation

X = [ones(m,1) X];  %5000x401

z2 = Theta1*X';  %(25x401)*(401x5000)
a2 = sigmoid(z2);  %25x5000

a2 = [ones(m,1) a2']; %5000x26

z3 = Theta2*a2'; %(10x26)*(26x5000)

h_theta = sigmoid(z3);  %h_theta = a3 the output 10x5000

 % y(k)  need to recode the labels as vectors containing only values 0 or 1 
y_new = zeros(num_labels,m); %10x5000

for i = 1:m
  y_new(y(i),i)=1;
end


J=(1/m)*sum(sum(-y_new.*log(h_theta)-(1-y_new).*log(1-h_theta)));

% Regularized cost function
Theta1_reg = Theta1(size(Theta1,1)+1:end); %Theta1_reg transforma Theta1 a un vector sin bias
Theta2_reg = Theta2(size(Theta2,1)+1:end); %Theta2_reg transforma Theta2 a un vector sin bias

R = (lambda/(2*m))*(sum(Theta1_reg(1:end).^2) + sum(Theta2_reg(1:end).^2));

J = J + R;

% Back propagation

for t = 1 : m  %5000
  % Step 1
  a1 = X(t,:); %vector con bias en a1(1) y de tama�o 1x401 
  
  z2 = Theta1*a1';  %(25x401)*(401x1)
  
  a2 = sigmoid(z2); %25x1
  
  a2 = [1; a2];  %26x1 
  
  z3 = Theta2*a2; %(10x26)*(26x1)
  
  a3 = sigmoid(z3); %10x1
  % Step 2
  delta_3 = a3 - y_new(:,t); %10x1
  
  z2 = [1; z2]; % bias 26x1
  % Step 3
  
  delta_2 = (Theta2'*delta_3).*sigmoidGradient(z2);  % (26x10)*(10x1).*26x1=26x1
  
  % Step 4
  delta_2 = delta_2(2:end); % skipping sigma2(0) (25*1)
  
  Theta2_grad = Theta2_grad + delta_3*a2'; %(10x1)*(1x26)
  Theta1_grad = Theta1_grad + delta_2*a1; %(25x1)*(1x401)
  
  
  
end

 % Step 5

 Theta2_grad = (1/m)*Theta2_grad; % 10x26
 Theta1_grad = (1/m)*Theta1_grad; % 25x401

% regularization gradients

 Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end); %for j>1

 Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end); %for j>1
 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
