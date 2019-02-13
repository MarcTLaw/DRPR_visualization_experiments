clear all;
close all;
hold all;

colormap('jet(32)');

labels = load('../stl10/labels.txt');
Y = load('../stl10_learned_representations/data_5_2_d.txt');

for i=0:9
    indices = (labels == i);    
    X = Y(indices,:);
    
    if i == 1
        colorused = [1,0,0];
    elseif i == 2
        colorused = [0,1,0];
    elseif i == 3
        colorused = [0,0,1];
    elseif i == 4
        colorused = [0.00000,0.39216,0.0000];
    elseif i == 5
        colorused = [1,0.84,0];
    elseif i == 6
        colorused = [0,1,1];
    elseif i == 7
        colorused = [0.5,0.5,0.5];
    elseif i == 8
        colorused = [1.00000,0.38824,0.27843];
    elseif i == 9
        colorused = [1.000000 ,0.078431,0.576471];
    elseif i == 0
        colorused = [0.58039,0,0.82745];
    end
        
    plot(X(:,1),X(:,2),'.','color',colorused);
    hold on
end
axis equal
axis off

legend('airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck')

