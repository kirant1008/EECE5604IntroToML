clc;
clear all;
figure(1)

no_components=1:1:6;
posteriror1=[-26.8185555903026,-26.7463826682505,-26.4190143591494,-27.3930084893671,-28.0939140406983,-27.6198739856497];
posteriror3=[-261.802623734611,-251.076081908914,-242.703430460785,-233.423597243256,-234.143099841457,-233.621738085792];
posteriror4=[-2653.02403292077,-2620.39404290378,-2545.10521395263,-2528.31215498735,-2512.59383977390,-2517.29850957623];
posteriror2=[-134.676852810389,-132.842136897282,-125.761300094579,-124.779377993363,-124.742626016661,-124.295381105688]
sgtitle('Plot of Likelihood for 100 500 1000 and 10000 samples')
subplot(2,1,1)
plotlikli(no_components,posteriror1);
subplot(2,1,2)
plotlikli(no_components,posteriror2);
figure(2)
subplot(2,1,1)
plotlikli(no_components,posteriror3);
subplot(2,1,2)
plotlikli(no_components,posteriror4);


function plotlikli(components,poster)
for i=1:6    
stem(components(i) ,poster(i));%for 100 Components
hold on
end
hold off
title('Plot of Likelihood of Each Component')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'C_1','C_2','C_3','C_4','C_5','C_6'},'Location','northeast')
end