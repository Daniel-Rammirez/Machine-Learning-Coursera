function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

%sum1=0;
%sum11=0;
% for i=1:m

 % sum1=sum1+(theta(1)*X(i,1)+theta(2)*X(i,2)-y(i))*X(i,1);
 % sum11=sum11+(theta(1)*X(i,1)+theta(2)*X(i,2)-y(i))*X(i,2);

%end
%theta(1)=theta(1)-(alpha/m)*sum1;
%theta(2)=theta(2)-(alpha/m)*sum11;

%J= (1/(2*m))* (X*theta-y)'*(X*theta-y);


%feature=size(X,2);
  
 % theta=theta-(alpha/m)*(X*theta-y)'*X;
 h = X * theta;
    errors = h - y;
    delta = X' * errors;
    theta = theta - (alpha / m) * delta;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
