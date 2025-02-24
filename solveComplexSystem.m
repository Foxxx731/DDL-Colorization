function X = solveComplexSystem(A, B)
    % Check if the input matrix is square
    [m, n] = size(A);
    if m ~= n
        error('Matrix A must be square');
    end
    
    % Check if A and B have compatible dimensions
    if size(B, 1) ~= m || size(B, 2) ~= 1
        error('Matrix B must be a column vector with the same number of rows as A');
    end
    
    % Check if the matrix is singular
    if det(A) == 0
        error('Matrix A is singular, system has no unique solution');
    end
    
    % Solve for X using matrix division
    X = A \ B;  % Equivalent to X = inv(A) * B but more efficient
    
    % Display the solution
    disp('Solution for X:');
    disp(X);
end
