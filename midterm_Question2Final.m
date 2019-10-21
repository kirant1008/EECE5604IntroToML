clc;
clear all;

x1=0;
y1=0;
rad_c=1;


%for generation of circle
for t=1:4
    [xt(t) yt(t)]=cir_with_draw(x1,y1,rad_c);
end


%K reference points
xk1=1;yk1=0;
xk2=[1,cos(pi)];yk2=[0,sin(pi)];
xk3=[1,cos((2*pi)/3),cos((4*pi)/3)];yk3=[0,sin((2*pi)/3),sin((4*pi)/3)];
xk4=[1,cos(pi/2),cos(pi),cos((3*pi)/2),cos(2*pi)];
yk4=[0,sin(pi/2),sin(pi),sin((3*pi)/2),sin(2*pi)];

%finding Distance
%case 1 k=1
for i=1:4
d1(i)=(sqrt((xt(i)-xk1)^2+(yt(i)-yk1)^2))%for 2 reference point,
end
d1=d1';
for i=1:4
    w1(i)=find_prob(1,1,xt(i),yt(i));
end
d1=cat(2,d1,[1:4]', w1');
%d1=[d1,[1;2;3;4]];

%case 2 k=2
for i=1:4
    for j=1:2
        d2(i,j)=(sqrt((xt(i)-xk2(j))^2+(yt(i)-yk2(j))^2))%for 2 reference point
    end
end
for i=1:4
    w2(i)=find_prob(1,1,xt(i),yt(i));
end
d2=cat(2,d2,[1:4]',w2');
% d2=[d2,[1;2;3;4]];


%case 3 k=3
for i=1:4
    for j=1:3
        d3(i,j)=(sqrt((xt(i)-xk3(j))^2+(yt(i)-yk3(j))^2))%for 3 reference point
    end
end
for i=1:4
    w3(i)=find_prob(1,1,xt(i),yt(i));
end
d3=cat(2,d3,[1:4]',w3');
% d3=[d3,[1;2;3;4]];
%case 4 k=4
for i=1:4
    for j=1:4
        d4(i,j)=(sqrt((xt(i)-xk4(j))^2+(yt(i)-yk4(j))^2))%for 4 reference point
    end
end
for i=1:4
    w4(i)=find_prob(1,1,xt(i),yt(i));
end
d4=cat(2,d4,[1:4]',w4');
% d4=[d4,[1;2;3;4]];

%noise generation
stand_dev=0.3;
%for case 1
n1 = -10 * ones(4, 1);
for i=1:4
    while n1(i,1)+d1(i,1)<0
        n1(i,1)=mvnrnd(0,stand_dev^2);
    end
end
%for case 2
n2 = -10 * ones(4, 2);
for i=1:4
    for j=1:2
         while n2(i,j)+d2(i,j)<0
            n2(i,j)=mvnrnd(0,stand_dev^2);
         end
    end
end
%for case 3
n3=-10*ones(4,3);
for i=1:4
    for j=1:3
        while n3(i,j)+d3(i,j)<0
         n3(i,j)=mvnrnd(0,stand_dev^2);
        end
    end
end
%for case 4
n4=-10*ones(4,4);
for i=1:4
    for j=1:4
        while n4(i,j)+d4(i,j)<0
          n4(i,j)=mvnrnd(0,stand_dev^2);
        end
    end
end


%%Generating ri
%case 1 r1 for 1 reference
r1=d1(:, 1)+n1;
%case 2 r2 for 2 reference
r2(:,1:2)=d2(:,1:2)+n2;
%case 3 r3 for 3 reference
r3(:,1:3)=d3(:,1:3)+n3;
%case 4 r4 for 4 refernce
r4(:,1:4)=d4(:,1:4)+n4;


%generating eucledian distance k=1
for i=1:4
    for j=1:4
        ed_case1(j,i)=norm(d1(j,1)-r1(i,1));
    end
end
ed_case1=ed_case1.^-1;
%generating eucledian distance k=2
for i=1:4
    for j=1:4
        ed_case2(j,i)=norm(d2(j,1:2)-r2(i,1:2));
    end
end
ed_case2=ed_case2.^-1;
%generating eucledian distance k=3
for i=1:4
    for j=1:4
        ed_case3(j,i)=norm(d3(j,1:3)-r3(i,1:3));
    end
end
ed_case3=ed_case3.^-1;
%generating eucledian distance k=4
for i=1:4
    for j=1:4
        ed_case4(j,i)=norm(d4(j,1:4)-r4(i,1:4));
    end
end
ed_case4=ed_case4.^-1;

%posterior prob for case 1 k=1
post_Case1=ed_case1.*d1(:,3);
%posterior prob for case 2 k=2
post_Case2=ed_case2.*d2(:,4);
%posterior prob for case 3 k=3
post_Case3=ed_case3.*d3(:,5);
%posterior prob for case 4 k=4
post_Case4=ed_case4.*d4(:,6);

%maximum of probablity
[max_k1,ind_k1]=max(post_Case1);
[max_k2,ind_k2]=max(post_Case2);
[max_k3,ind_k3]=max(post_Case3);
[max_k4,ind_k4]=max(post_Case4);







figure(1)

sgtitle('Plot of True Point and K reference Points')

subplot(2,2,1)

plot(xt,yt,'o')
hold on
draw_circle(x1,y1,rad_c);
hold on
plot(xk1,yk1,'X','color','b','MarkerSize',10)
hold off


xlabel('X-Coordiante')
ylabel('Y-Coordinate')
legend({'True Position','K Reference'},'Location','northeast')

subplot(2,2,2)
plot(xt,yt,'o')
hold on
draw_circle(x1,y1,rad_c);
hold on
plot(xk2,yk2,'X','color','b','MarkerSize',10)
hold off

xlabel('X-Coordiante')
ylabel('Y-Coordinate')
legend({'True Position','K Reference'},'Location','northeast')

subplot(2,2,3)
plot(xt,yt,'o')
hold on
draw_circle(x1,y1,rad_c);
hold on
plot(xk3,yk3,'X','color','b','MarkerSize',10)
hold off

xlabel('X-Coordiante')
ylabel('Y-Coordinate')
legend({'True Position','K Reference'},'Location','northeast')


subplot(2,2,4)
plot(xt,yt,'o')
hold on
draw_circle(x1,y1,rad_c);
hold on
plot(xk4,yk4,'x','color','b','MarkerSize',10)
hold off

xlabel('X-Coordiante')
ylabel('Y-Coordinate')
legend({'True Position','K Reference'},'Location','northeast')

figure(2)

%for contours

subplot(2,2,1)
contour(post_Case1)
title('Plot of Contour for 1 reference Point');
subplot(2,2,2)
contour(post_Case2)
title('Plot of Contour for 2 reference Points');
subplot(2,2,3)
contour(post_Case3)
title('Plot of Contour for 3 reference Points');
subplot(2,2,4)
contour(post_Case4)
title('Plot of Contour for 4 reference Points');

function [x y]=cir_with_draw(x1,y1,r_c)
a=2*pi*rand;
r=-1+2*rand;
x=(r_c*r)*cos(a)+x1;
y=(r_c*r)*sin(a)+y1;
end

function draw_circle(x1,y1,radius)
viscircles([x1,y1],radius)
end

function dist = find_dist(xk,yk,xt,yt)
dist =sqrt((xk-xt)^2+(yk-yt)^2);
end

function prob_xy = find_prob(sigx,sigy,x,y)
prob_xy=mvnpdf([x y],[0 0],[sigx^2 0;0 sigy^2]);
end