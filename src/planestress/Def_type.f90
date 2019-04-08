
! ---------------------------------------------------------------------------------------
!            Module for defining user-defined types
!            - created by Phill-Seung Lee, 19/Feb/1998 
!            - FORTRAN 95
!
!            * Note: This source code can be freely used, distributed and modified 
!                    only for academic purposes.
! ---------------------------------------------------------------------------------------
 
module Def_type  ! -------------- Begin of Module ---------------------------------------

   ! define a type for node
   type :: NodeType
      double precision :: x(3)       ! nodal position (x, y, z)
      double precision :: pm(6)      ! nodal force (Px, Py, Pz, Mx, My, Mz)
      integer          :: bc(6)      ! displacement BC (u, v, w, Rx, Ry, Rz) (1=fixed, 0=free)
      integer          :: eq_n(6)    ! equation number (u, v, w, Rx, Ry, Rz)
   end type NodeType

   ! define a type for element 
   type :: ElementType
      integer          :: cn(4)      ! connectivity
      double precision :: thickness  ! thickness
      double precision :: q(2)       ! distributed load in x- and y-directions
   end type ElementType

   ! define a type for material
   type :: MaterialType
      double precision :: Young      ! Young's modulus
      double precision :: Poisson    ! Poison's ratio
   end type MaterialType 

end module Def_type  ! --------------- End of Module ----------------------------------
