X = load('original_data.txt');
temperatures = [1,2,3,4,5,10,50];
for temperature = temperatures
    A = exp(X ./ temperature);
    A = bsxfun(@rdivide, A, sum(A,2));
    save(strcat('proba_data_',int2str(temperature),'.txt'),'A','-ascii')
end
