
! ---------------------------------------------------------------------------------------
!            Module for skyline solver to solve linear equations 
!            - copied from a textbook "Finite Element Procedures"
!            - modified by Phill-Seung Lee
!            - Fortran 95
! ---------------------------------------------------------------------------------------

module Solver  ! -------------- Begin of Module -----------------------------------------

   public COLSOL

contains  ! ------------------------------ Internal Procedures --------------------------

! ---------------------------------------------------------------------------------------

   subroutine COLSOL(A,V,MAXA,NN,NWK,NNM,KKK,IOUT)
! . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
! .                                                                   . 
! .   P R O G R A M                                                   . 
! .        TO SOLVE FINITE ELEMENT STATIC EQUILIBRIUM EQUATIONS IN    . 
! .        CORE, USING COMPACTED STORAGE AND COLUMN REDUCTION SCHEME  . 
! .                                                                   . 
! .  - - INPUT VARIABLES - -                                          . 
! .        A(NWK)    = STIFFNESS MATRIX STORED IN COMPACTED FORM      . 
! .        V(NN)     = RIGHT-HAND-SIDE LOAD VECTOR                    . 
! .        MAXA(NNM) = VECTOR CONTAINING ADDRESSES OF DIAGONAL        . 
! .                    ELEMENTS OF STIFFNESS MATRIX IN A              . 
! .        NN        = NUMBER OF EQUATIONS                            . 
! .        NWK       = NUMBER OF ELEMENTS BELOW SKYLINE OF MATRIX     . 
! .        NNM       = NN + 1                                         . 
! .        KKK       = INPUT FLAG                                     . 
! .            EQ. 1   TRIANGULARIZATION OF STIFFNESS MATRIX          . 
! .            EQ. 2   REDUCTION AND BACK-SUBSTITUTION OF LOAD VECTOR . 
! .        IOUT      = UNIT NUMBER USED FOR OUTPUT                    . 
! .                                                                   . 
! .  - - OUTPUT - -                                                   . 
! .        A(NWK)    = D AND L - FACTORS OF STIFFNESS MATRIX          . 
! .        V(NN)     = DISPLACEMENT VECTOR                            . 
! .                                                                   . 
! . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)                               
! . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
! .   THIS PROGRAM IS USED IN SINGLE PRECISION ARITHMETIC ON CRAY     . 
! .   EQUIPMENT AND DOUBLE PRECISION ARITHMETIC ON IBM MACHINES,      . 
! .   ENGINEERING WORKSTATIONS AND PCS. DEACTIVATE ABOVE LINE FOR     . 
! .   SINGLE PRECISION ARITHMETIC.                                    . 
! . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
      double precision, intent(inout) :: A(NWK), V(NN)
      integer,          intent(in)    :: MAXA(NNM)                                  
!                                                                       
!     PERFORM L*D*L(T) FACTORIZATION OF STIFFNESS MATRIX                
!                                                                       
      IF (KKK-2) 40,150,150                                             
   40 DO 140 N=1,NN                                                     
      KN=MAXA(N)                                                        
      KL=KN + 1                                                         
      KU=MAXA(N+1) - 1                                                  
      KH=KU - KL                                                        
      IF (KH) 110,90,50                                                 
   50 K=N - KH                                                          
      IC=0                                                              
      KLT=KU                                                            
      DO 80 J=1,KH                                                      
      IC=IC + 1                                                         
      KLT=KLT - 1                                                       
      KI=MAXA(K)                                                        
      ND=MAXA(K+1) - KI - 1                                             
      IF (ND) 80,80,60                                                  
   60 KK=MIN0(IC,ND)                                                    
      C=0.                                                              
      DO 70 L=1,KK                                                      
   70 C=C + A(KI+L)*A(KLT+L)                                            
      A(KLT)=A(KLT) - C                                                 
   80 K=K + 1                                                           
   90 K=N                                                               
      B=0.                                                              
      DO 100 KK=KL,KU                                                   
      K=K - 1                                                           
      KI=MAXA(K)                                                        
      C=A(KK)/A(KI)                                                     
      B=B + C*A(KK)                                                     
  100 A(KK)=C                                                           
      A(KN)=A(KN) - B                                                   
  110 IF (A(KN)) 120,120,140                                            
  120 WRITE (iout,2000) N,A(KN)
      WRITE (*,2000) N,A(KN)
      pause                              
      GO TO 800                                                         
  140 CONTINUE                                                          
      GO TO 900                                                         
!                                                                       
!     REDUCE RIGHT-HAND-SIDE LOAD VECTOR                                
!                                                                       
  150 DO 180 N=1,NN                                                     
      KL=MAXA(N) + 1                                                    
      KU=MAXA(N+1) - 1                                                  
      IF (KU-KL) 180,160,160                                            
  160 K=N                                                               
      C=0.                                                              
      DO 170 KK=KL,KU                                                   
      K=K - 1                                                           
  170 C=C + A(KK)*V(K)                                                  
      V(N)=V(N) - C                                                     
  180 CONTINUE                                                          
!                                                                       
!     BACK-SUBSTITUTE                                                   
!                                                                       
      DO 200 N=1,NN                                                     
      K=MAXA(N)                                                         
  200 V(N)=V(N)/A(K)                                                    
      IF (NN.EQ.1) GO TO 900                                            
      N=NN                                                              
      DO 230 L=2,NN                                                     
      KL=MAXA(N) + 1                                                    
      KU=MAXA(N+1) - 1                                                  
      IF (KU-KL) 230,210,210                                            
  210 K=N                                                               
      DO 220 KK=KL,KU                                                   
      K=K - 1                                                           
  220 V(K)=V(K) - A(KK)*V(N)                                            
  230 N=N - 1                                                           
      GO TO 900                                                         
!                                                                       
  800 STOP                                                              
  900 RETURN                                                            
!                                                                       
 2000 FORMAT (//' STOP - STIFFNESS MATRIX NOT POSITIVE DEFINITE',//,  &  
                ' NONPOSITIVE PIVOT FOR EQUATION ',I8,//,             &  
                ' PIVOT = ',E20.12 )                                    
 
   end subroutine COLSOL

! ---------------------------------------------------------------------------------------

end module Solver  ! --------------- End of Module --------------------------------------