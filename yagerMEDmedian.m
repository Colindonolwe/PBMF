beta = 0.6;
betal = beta;
betap = beta;
betar = beta;
betag = beta;

ig = 2;
ip = 3;
ir = 7;
il = 8;

Ag = 0.8;
Ap = 0.6;
Ar = 0.4;
Al = 0.2;


% Ag = 0;
% Ap = 0;
% Ar = 0;
% Al = 0;
% for Ag=0.2:0.2:0.8
%    for Ap=0.2:0.2:0.8
%         for Ar=0.2:0.2:0.8
%             for Al=0.2:0.2:0.8
% for ig=1:1:6
%    for ip=ig+1:1:7
%         for ir=ip+1:1:8
%             for il=ir+1:1:9
%                
% figure;
% imshow(A);
% names = ['noisyImageImp20.jpg','noisyImageImp30.jpg','noisyImageGaus20.jpg','noisyImageGaus30.jpg','noisyImageGaus20Imp20.jpg','noisyImageImp50.jpg','noisyImageSP20.jpg'];
% names = ['noisyImageImp20.jpg','noisyImageGaus20.jpg','noisyImageGaus20Imp20.jpg','noisyImageSP20.jpg'];
% data = ['noisyImageImp20.jpg'; 'noisyImageImp30.jpg'];
% names = cellstr(data);
for ni=1:1:1
    J2 = impulsenoise(Agr1,0.2,1);
%     A = im2double(imread(char(names(ni))));
    A = im2double(J2);
          flg = 0;
            for rk=1:1:10
%                 for beta=0.5:0.1:0.8
%             betal = beta;
%             betap = beta;
%             betar = beta;
%             betag = beta;
            ftrN = zeros(9,m,n);
            ftrF = zeros(9,m,n);
            ftrNF = zeros(9,m,n); 

                mmm = zeros(m,n);
                mN = zeros(m,n);
                mF = zeros(m,n);
                mNF = zeros(m,n);
                mBel = 30;
                filt = zeros(mBel,m,n);
            
                for i=2:m-1
                    for j=2:n-1
                        if j ~=2 && flg
                           A(i,j-1) = yout(i,j-1); 
                        end
                            Med = [];
                            dA = [];
                            Med(1) = A(i-1,j-1);
                            Med(2) = A(i-1,j) ;
                            Med(3) = A(i-1,j+1);
                            Med(4) = A(i,j-1);
                            Med(5) = A(i,j+1);
                            Med(6) = A(i+1, j-1);
                            Med(7) = A(i+1,j);
                            Med(8) = A(i+1,j+1);
                            
                            med_sorted = sort(Med);
                            M = size(med_sorted);
                            dA(1) = abs(A(i,j)-A(i-1,j-1));
                            dA(2) = abs(A(i,j)-A(i-1,j));
                            dA(3) = abs(A(i,j)-A(i-1,j+1));
                            dA(4) = abs(A(i,j)-A(i,j-1));
                            dA(5) = abs(A(i,j)-A(i,j+1));
                            dA(6) = abs(A(i,j)-A(i+1, j-1));
                            dA(7) = abs(A(i,j)-A(i+1,j));
                            dA(8) = abs(A(i,j)-A(i+1,j+1));
                            d_sorted = sort(dA);
                            aAvg = mean(Med);
                            
%                             ftrN(1,i,j) =  (1-Ag)*abs(A(i,j)-aAvg);
%                             ftrNF(1,i,j) = Ag+(1-Ag)*betag*(1-ftrN(1,i,j));
%                             ftrF(1,i,j) = (1-Ag)*(1-betag)*(1-ftrN(1,i,j));
                            
                            Med(9) = A(i,j);
                            aMean = mean(Med);
                            mmm(i,j) = median(Med);
                            sumMed = 0;
                            for k=1:1:9
                               sumMed = sumMed + abs(Med(k)-aMean);
                            end
                            ftrN(2,i,j) = (1-Ag)*abs(A(i,j)-aMean)/sumMed;
                            ftrNF(2,i,j) = Ag+(1-Ag)*betap*(1-ftrN(2,i,j));
                            ftrF(2,i,j) = (1-Ag)*(1-betap)*(1-ftrN(2,i,j));
                            
                            sumd = 0;
                            for k=1:1:4
                               sumd = sumd + d_sorted(k);
                            end
                            ftrN(3,i,j) = (1-Ap)*sumd/4;
                            ftrNF(3,i,j) = Ap+(1-Ap)*betap*(1-ftrN(3,i,j));
                            ftrF(3,i,j) = (1-Ap)*(1-betap)*(1-ftrN(3,i,j));
                            
%                             ftrN(4,i,j) = (1-Ap)*(d_sorted(1)+d_sorted(2))/2;
%                             ftrNF(4,i,j) = Ap+(1-Ap)*betal*(1-ftrN(4,i,j));
%                             ftrF(4,i,j) = (1-Ap)*(1-betal)*(1-ftrN(4,i,j));
                            
%                             c(9) = abs(A(i,j)-aMean)/sumMed;
%                             c(1) = abs(c(9) - abs(A(i-1,j-1)-aMean)/sumMed);
%                             c(2) = abs(c(9) - abs(A(i-1,j)-aMean)/sumMed);
%                             c(3) = abs(c(9) - abs(A(i-1,j+1)-aMean)/sumMed);
%                             c(4) = abs(c(9) - abs(A(i,j-1)-aMean)/sumMed);
%                             c(5) = abs(c(9) - abs(A(i,j+1)-aMean)/sumMed);
%                             c(6) = abs(c(9) - abs(A(i+1,j-1)-aMean)/sumMed);
%                             c(7) = abs(c(9) - abs(A(i+1,j)-aMean)/sumMed);
%                             c(8) = abs(c(9) - abs(A(i+1,j+1)-aMean)/sumMed);
% % %                             
%                             c_sorted = sort(c(1:8));
%                             ftrN(5,i,j) = (1-Ar)*(c_sorted(1)+c_sorted(2))/2;
%                             ftrNF(5,i,j) = Ar+(1-Ar)*betap*(1-ftrN(5,i,j));
%                             ftrF(5,i,j) = (1-Ar)*(1-betap)*(1-ftrN(5,i,j));
% %                             
% %                             
%                             aAvg = median(Med);
%                             ftrN(6,i,j) =  (1-Ar)*abs(A(i,j)-aAvg);
%                             ftrNF(6,i,j) = Ar+(1-Ar)*betag*(1-ftrN(6,i,j));
%                             ftrF(6,i,j) = (1-Ar)*(1-betag)*(1-ftrN(6,i,j));
%                             
                            Med(10) = A(i,j);
                            Med(11) = A(i,j);
%                             
                            aAvg = median(Med);
                            ftrN(7,i,j) =  (1-Ar)*abs(A(i,j)-aAvg);
                            ftrNF(7,i,j) = Ar+(1-Ar)*betag*(1-ftrN(7,i,j));
                            ftrF(7,i,j) = (1-Ar)*(1-betag)*(1-ftrN(7,i,j));
                            
                            Med(12) = A(i,j);
                            Med(13) = A(i,j);
                            
                            aAvg = median(Med);
                            ftrN(8,i,j) =  (1-Al)*abs(A(i,j)-aAvg);
                            ftrNF(8,i,j) = Al+(1-Al)*betar*(1-ftrN(8,i,j));
                            ftrF(8,i,j) = (1-Al)*(1-betar)*(1-ftrN(8,i,j));
%                             
%                             ind = uint8(M(2)*1/2);
%                             ftrN(9,i,j) = (1-Al)*abs(A(i,j) - (med_sorted(ind-1)+med_sorted(ind))/2);
%                             ftrNF(9,i,j) = Al+(1-Al)*betal*(1-ftrN(9,i,j));
%                             ftrF(9,i,j) = (1-Al)*(1-betal)*(1-ftrN(9,i,j));
%                             
                                mN(i,j) = ftrN(ig,i,j)*ftrN(ir,i,j)*ftrN(il,i,j)*ftrN(ip,i,j)+ftrNF(ig,i,j)*ftrN(ir,i,j)*ftrN(il,i,j)*ftrN(ip,i,j)...
                                +ftrN(ig,i,j)*ftrNF(ir,i,j)*ftrN(il,i,j)*ftrN(ip,i,j)...
                                +ftrN(ig,i,j)*ftrN(ir,i,j)*ftrNF(il,i,j)*ftrN(ip,i,j)+ftrN(ig,i,j)*ftrN(ir,i,j)*ftrN(il,i,j)*ftrNF(ip,i,j)+ftrNF(ig,i,j)*ftrNF(ir,i,j)*ftrN(il,i,j)*ftrN(ip,i,j)...
                                +ftrNF(ig,i,j)*ftrN(ir,i,j)*ftrNF(il,i,j)*ftrN(ip,i,j)+ftrNF(ig,i,j)*ftrN(ir,i,j)*ftrN(il,i,j)*ftrNF(ip,i,j)+ftrN(ig,i,j)*ftrNF(ir,i,j)*ftrNF(il,i,j)*ftrN(ip,i,j)...
                                +ftrN(ig,i,j)*ftrNF(ir,i,j)*ftrN(il,i,j)*ftrNF(ip,i,j)+ftrN(ig,i,j)*ftrN(ir,i,j)*ftrNF(il,i,j)*ftrNF(ip,i,j)+ftrNF(ig,i,j)*ftrNF(ir,i,j)*ftrNF(il,i,j)*ftrN(ip,i,j)...
                                +ftrNF(ig,i,j)*ftrNF(ir,i,j)*ftrN(il,i,j)*ftrNF(ip,i,j)+ftrNF(ig,i,j)*ftrN(ir,i,j)*ftrNF(il,i,j)*ftrNF(ip,i,j)+ftrN(ig,i,j)*ftrNF(ir,i,j)*ftrNF(il,i,j)*ftrNF(ip,i,j);

                                 
                    end
                end
%                 mN=pN;
                
                curT = zeros(mBel,1);
                histmN = histogram(mN,mBel-1);
                bound = histmN.BinEdges';
%                 ymy = zeros(m,n);
%                 bound = 0:1/mBel:1;
                for b=1:1:size(bound)
                    for i=2:m-1
                        for j=2:n-1
                            if bound(b) < mN(i,j) && bound(b+1) > mN(i,j)
                                filt(b,i,j) = mN(i,j)+1;
                                curT(b) = curT(b)+1;
                            end
%                             if mN(i,j) >0.2 && mN(i,j) < 0.97
%                                 ymy(i,j) = mN(i,j);
%                             else
%                                 ymy(i,j) = A(i,j);
%                             end
                        end
                    end
                end
                % psnr(A,dtr)
%                 psnrEr = psnr(ymy,dtr)
                
alphas = zeros(mBel,1);
sumMSE = 0;
MAE = 0;
alphat = 1;
for ii = 1:1:mBel
    d = zeros(curT(ii),1);
    bel = zeros(curT(ii),1);
    x = zeros(curT(ii),1);
    med = zeros(curT(ii),1);
    k = 1;
    for i=2:m-1
        for j=2:n-1
            if filt(ii,i,j) > 0
                d(k) = dtr(i,j);
                bel(k) = filt(ii,i,j) - 1;
                x(k) = A(i,j);
                med(k) = mmm(i,j);
                k = k+1;
            end
        end
    end
    T = curT(ii);
%     alpha = 1/mBel*(mBel+1-ii);
    alpha = round(alphat,2); 
    alpha0=0.09;
%     alpha = alpha0;
    MSE = 0;
        for igr=1:1:size(x)
            y = med(igr) + alpha*(x(igr) - med(igr));
            eta = alpha0*(1-(igr/T));  
            alphat = alpha;
            alpha = alphat + 2*eta*(d(igr)-y)*(x(igr)-med(igr));

            MSEt = MSE;
            MSE = MSE + (255^2)*(d(igr)-y)*(d(igr)-y);
            if MSE ~= 0 && MSEt ~= 0 && (abs(MSE-MSEt)/MSE < 0.000000000001)
               sumMSE = sumMSE + MSE;
               break; 
            end
        end
    alphas(ii) = round(alphat,2);
end
% 
% MAE2=0;
yout = zeros(m,n);
for ii = 1:1:mBel
    for i=2:m-1
        for j=2:n-1
            if filt(ii,i,j) > 0
                fVal = filt(ii,i,j) - 1;
                yout(i,j) = mmm(i,j) + alphas(ii)*(A(i,j) - mmm(i,j));
                if yout(i,j)>1
                    yout(i,j) = 1;
                end
                if yout(i,j) < 0
                    yout(i,j) = 0;
                end
                MAE = MAE + abs(dtr(i,j)-yout(i,j));
            end
        end
    end
end

            flg = 1;
%             imshow(yout);

               end
                
% imshow(yout);
%                 end
%             beta

            
%             sumMSE
            
            255*MAE/(m*n)
            psnr(yout,dtr)
            ssim(yout,dtr)
%             imshow(yout);
%             psnrEr(ig, ip, ir, il)= psnr(yout,dtr);
%             maeEr(ig, ip, ir, il) = 255*MAE/(m*n);
%             mse(ig, ip, ir, il) = sumMSE;
            A = im2double(J);
end
%             mse(uint8(5*Ag),uint8(5*Ap),uint8(5*Ar),uint8(5*Al)) = sumMSE;
%             maeEr(uint8(5*Ag),uint8(5*Ap),uint8(5*Ar),uint8(5*Al)) = 255*MAE/(m*n);
%             psnrEr(uint8(5*Ag),uint8(5*Ap),uint8(5*Ar),uint8(5*Al)) = psnr(yout,dtr);
            
% 
%             end     
%         end
%     end
% end  

