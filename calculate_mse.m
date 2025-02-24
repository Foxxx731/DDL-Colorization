function mse = calculate_mse(image1, image2)
    % Ensure both images have the same dimensions
    if ~isequal(size(image1), size(image2))
        error('Images must have the same dimensions.');
    end

    % Convert images to double precision for accurate computation
    image1 = double(image1);
    image2 = double(image2);

    % Calculate the Mean Squared Error
    mse = mean((image1(:) - image2(:)).^2);
end

