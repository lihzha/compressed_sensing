img_path = 'C:\Users\CharlieHan\Desktop\bigtiger.jpeg';
img = imread(img_path);
img_re = imresize(img, [50 NaN], 'bilinear');
img_re=double(img_re);
[r_num,c_num,d_num] = size(img_re);
result = zeros(r_num, c_num, d_num);
k = floor(r_num * c_num * pct);
ri = randperm(r_num * c_num, k);
lumda = 100;

% Array-To-Integer Mapping
img_2d = ones(r_num,c_num);
for i = 1:r_num
    for j = 1:c_num
        img_2d(i,j) = img_re(i,j,1) + 256 * img_re(i,j,2) + 256^2 * img_re(i,j,3);
    end
end
img_1d = img_2d(:);
b = img_1d(ri);
A = kron(idct(eye(c_num)), idct(eye(r_num)));
A = A(ri, :);
cvx_begin
    variable x(r_num*c_num)
    minimize( norm( A * x - b, 2 )+ lumda * norm( x, 1) )
    subject to
        A * x == b
cvx_end
x = reshape(x, r_num, c_num);
x_a = idct2(x);
x_at = zeros(r_num, c_num, d_num);
for i = 1:r_num
    for j = 1:c_num
        x_at(i,j,3) = floor(x_a(i,j)/256/256);
        x_at(i,j,2) = floor((x_a(i,j)-x_at(i,j,3)*256*256)/256);
        x_at(i,j,1) = x_a(i,j)-x_at(i,j,3)*256*256-x_at(i,j,2)*256;
    end
end
x_at = uint8(x_at);
for i = 1:r_num
    for j = 1:c_num
        for k = 1:d_num
            if x_at(i,j,k)==0
                x_at(i,j,k)=255;
            end
        end
    end
end

% Separatiion of channels
img_num = 2;
results = zeros(r_num, c_num, d_num, img_num);
percentage = [0.01, 0.1,0.5];
for j = 1:img_num
    for i = 1:d_num
        img_1d = img_re(:,:,i);  
        img_1d = img_1d(:);
        pct = percentage(j);
        k = floor(r_num * c_num * pct);
        ri = randperm(r_num * c_num, k);
        b = img_1d(ri);
        A = kron(idct(eye(c_num)), idct(eye(r_num)));
        A = A(ri, :);
        cvx_begin
            variable x(r_num*c_num)
            minimize( norm( A * x - b, 2 )+ lumda * norm( x, 1) )
            subject to
                A * x == b
        cvx_end
        x = reshape(x, r_num, c_num);
        x_a = idct2(x);
        result(:,:,i) = x_a;
    end
    result = uint8(result);
    results(:,:,:,j) = result;
end

img_re = uint8(img_re);
results(:,:,:,3) = aaa;
results = uint8(results);

subplot(1,4,1);
imshow(img_re);
title('Downsized Image','fontsize',16,'interpreter','latex')
subplot(1,4,2);
imshow(results(:,:,:,1));
title('Percentage = 1\%','fontsize',16,'interpreter','latex')
subplot(1,4,3);
imshow(results(:,:,:,2));
title('Percentage = 10\%','fontsize',16,'interpreter','latex')
subplot(1,4,4);
imshow(results(:,:,:,3));
title('Percentage = 50\%','fontsize',16,'interpreter','latex')