% Load two images (ensure they have the same dimensions)
colorized_image = imread('Colorized_Dog_eccv16.png');
realistic_image = imread('Dog_norm.jpg');

% Resize the images to the same dimensions, if necessary
% realistic_image = imresize(realistic_image, size(colorized_image));

% Call the MSE function
mse_value = calculate_mse(colorized_image, realistic_image);

fprintf('Mean Squared Error (MSE): %.4f\n', mse_value);
