clear all;
close all;
hold all;

colormap('jet(32)');

labels = load('../cifar100/labels.txt');
Y = load('../cifar100_learned_representations/data_4_2_d.txt');

for i=1:20
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
        colorused = [255,178,102]/255;
    elseif i == 10
        colorused = [178,102,255]/255;
    elseif i == 11
        colorused = [255,0,127]/255;
    elseif i == 12
        colorused = [102,102,0]/255;
    elseif i == 13
        colorused = [153,255,204]/255;
    elseif i == 14
        colorused = [0,51,102]/255;
    elseif i == 15
        colorused = [153,0,153]/255;
    elseif i == 16
        colorused = [153,76,0]/255;
    elseif i == 17
        colorused = [255,204,229]/255;
    elseif i == 18
        colorused = [64,64,64]/255;
    elseif i == 19
        colorused = [0,153,153]/255;
    elseif i == 20
        colorused = [102,204,0]/255;
    end
        
    plot(X(:,1),X(:,2),'.','color',colorused);
    hold on
end
axis equal
axis off

legend('aquatic mammals', 'fish','flowers','food containers','fruit and vegetables','household electrical devices', 'household furniture','insects','large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores', 'medium mammals', 'non-insect invertebrates', 'people','reptiles', 'small mammals','trees', 'vehicles 1','vehicles 2');
