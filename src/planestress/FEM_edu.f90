
! ---------------------------------------------------------------------------------------
!            FEM code for educational purposes, OSE532, KAIST		  
!            - originally created by Phill-Seung Lee, 19/Feb/1998
!            - modified for educational purposes, Fall 2006, Fall 2009, Fall 2010 
!            - FORTRAN 95
!
!            - Elements : 4-node 2D plane stress element
!           
!            * Note: This source code can be freely used, distributed and modified 
!                    only for academic purposes. 
! ---------------------------------------------------------------------------------------

program FEM_edu  ! ------------------------- Begin of Program ---------------------------
   
   use Def_type
   use PlaneS
   use Solver

   implicit none
  
   ! define internal variables
   integer :: node_n, element_n                             ! # of nodes, # of elements
   type (NodeType),    allocatable, dimension(:) :: Node    ! node
   type (ElementType), allocatable, dimension(:) :: Element ! element
   type (MaterialType) :: Material                          ! material

   integer :: total_dof_n, free_dof_n, fixed_dof_n          ! # of DOFs (total, free, fixed)
   double precision, allocatable, dimension(:) :: Kt        ! stiffness vector
   double precision, allocatable, dimension(:) :: U         ! diplacement vector
   double precision, allocatable, dimension(:) :: R         ! load vector
   integer,          allocatable, dimension(:) :: c_index   ! column index

   call Main ! main subroutine

contains  ! ------------------------------ Internal Procedures --------------------------

! ---------------------------------------------------------------------------------------

! main subroutine
subroutine Main()
   character(len=20) :: filename
   
   ! input file name
   call Print_Title(0)
   write(*, "(' 1/ Input File Name (.txt): ',$)")
   read(*,*) filename
   
   ! open files
   open(unit=1, file=trim(filename)//".txt",     form="formatted")
   open(unit=2, file=trim(filename)//"_out.txt", form="formatted")
   open(unit=3, file=trim(filename)//"_res.txt", form="formatted")
   open(unit=4, file=trim(filename)//"_pos.txt", form="formatted")
   call Print_Title(3)

   ! read input file    
   print *, "2/ Reading Input File";
   call Read_Input_File
   
   ! calculate # of total DOF, # of free DOF, # of fixed DOF
   ! and assign equation numbers
   call Set_DOF_Number(total_dof_n, free_dof_n, fixed_dof_n)
   
   ! calculate column index
   call Calculate_Index
   
   ! allocate memory
   allocate( Kt( c_index(free_dof_n+1)-1 ) ) ! total stiffness vector (Kt)
   allocate( U(total_dof_n) )                ! displacement vector
   allocate( R(total_dof_n) )                ! load vector
   
   ! assemble total stiffness matrix
   print *, "3/ Assembling Stiffness and Load"
   call Assemble_Kt

   ! assemble load vector
   call Assemble_Load

   ! slove linear system (find U in K*U=R)
   print *, "4/ Solving Linear System"
   call Solve_Equations 

   ! calculate stress and print solutions
   print *, "5/ Printing Output Files"
   call Displacement_Stress

   ! deallocate memory
   deallocate(Node, Element, Kt, R, U, c_index)

   ! close files
   close(unit=1); close(unit=2); close(unit=3); close(unit=4)
   print *, "6/ Completed !"; print *, ""; pause
end subroutine Main

! ---------------------------------------------------------------------------------------
 
! read input file
subroutine Read_Input_File()
   double precision :: thickness
   character :: bufs
   integer :: bufi, i

   ! read nodal information
   read(1,*) bufs; read(1,*) node_n; read(1,*) bufs
   allocate(Node(node_n)) ! allocate a node array  
   do i=1, node_n
      read(1,*) bufi, Node(i).x(1:2), Node(i).bc(1:2), Node(i).pm(1:2)
      Node(i).bc(3:6) = 1 ! fixed DOFs not used in 2D plane stress problems.
   end do

   ! read elemental information
   read(1,*) bufs; read(1,*) element_n; read(1,*) bufs
   allocate(Element(element_n)) ! allocate a element array
   do i=1, element_n
      read(1,*) bufi, Element(i).cn(:), Element(i).q(:)
   end do

   ! read properties
   read(1,*) bufs; read(1,*) thickness
   Element(:).thickness = thickness
   read(1,*) bufs; read(1,*) Material.Young
   read(1,*) bufs; read(1,*) Material.Poisson
end subroutine Read_Input_File

! ---------------------------------------------------------------------------------------

! calculate # of total DOF, # of free DOF, # of fixed DOF
! and assign equation numbers to DOFs
subroutine Set_DOF_Number(tn, fn, cn)
   integer, intent(out) :: tn, fn, cn  ! # of total DOF, # of free DOF, # of fixed DOF
   integer :: i, j
   
   write(3, *) "EQUATION NUMBER"
   write(3, *) "---------------"
   write(3, *) "    node   dof    eqn"
   
   tn = node_n * 6  ! # of total DOF
   fn = 0; cn = 0
   do i=1, node_n
      do j=1, 6
         if (Node(i).bc(j) == 0) then 
            fn = fn + 1
            Node(i).eq_n(j) = fn
            write(3, "(3i7)") i, j, fn
         else
            cn = cn + 1
            Node(i).eq_n(j) = tn - cn + 1
         end if
      end do
   end do
   write(3, *)      
end subroutine Set_DOF_Number

! ---------------------------------------------------------------------------------------

! calculate column index for skyline solver
subroutine Calculate_Index()   
   integer :: column_h(total_dof_n) ! column height
   integer :: a_index(6*4)          ! index for assemblage
   integer :: en                    ! number of element nodes
   integer :: i, j, k               ! index for iteration
   integer :: buf_sum

   ! allocate c_index array
   allocate ( c_index(free_dof_n+1) ) 
   
   ! column height
   column_h(:) = 0
   do i=1, element_n
      ! assemblage index
      en = 4
      do j=1, 6
         do k=1, en
           a_index( en*j+k-en ) = node( element(i).cn(k) ).eq_n(j)
         end do
      end do      
      ! column height   
      do k=1, 6*en
         do j=1, 6*en
            if(a_index(j)<=free_dof_n .and. a_index(k)<=free_dof_n) then
               if( a_index(j) < a_index(k) ) then
                  if( a_index(k)-a_index(j) > column_h(a_index(k)) ) then
                     column_h(a_index(k)) = a_index(k) - a_index(j)
                  end if
               end if
            end if
         end do
      end do  
   end do

   ! c_index array
   buf_sum = 1
   do i=1, free_dof_n
      c_index(i) = buf_sum
      buf_sum = buf_sum + Column_H(i) + 1
   end do
   c_index(free_dof_n+1) = buf_sum
   write(3,"(a18, i4, a3)") "REQUIRED MEMORY =", buf_sum*8/1000000, " MB"
   write(3,*)
end subroutine Calculate_Index

! ---------------------------------------------------------------------------------------

! assemble total stiffness matrix by using equation numbers
subroutine Assemble_Kt()
   double precision :: eNode(4,2) ! nodal position of 4-node element (x,y)
   double precision :: Ke(8,8)    ! stifness matrix of element
   integer :: a_index(8)          ! assemblage index
   integer :: i, j, k, address

   Kt(:) = 0.0d0
  
   do i=1, element_n
      ! nodal position of element
      do j=1, 4
        do k=1, 2
           eNode(j, k) = Node( Element(i).cn(j) ).x(k)
        end do
      end do  
      ! calculate stiffness matrix of element
      call Plane_Stiffness(Material.Young, Material.Poisson, Element(i).thickness, eNode, Ke)
      write(2,"(a24,i4)") " STIFFNESS of ELEMENT : ", i
      call Print_Matrix(Ke)      
      ! assemblage index
      do j=1, 2
         do k=1, 4
            a_index( 4*j+k-4 ) = node( element(i).cn(k) ).eq_n(j)
         end do
      end do
      ! assemble total stiffness matrix
      do j=1, 8
         do k=1, 8
            if(a_index(j)<=free_dof_n .and. a_index(k)<=free_dof_n) then
               if( a_index(j) <= a_index(k) ) then
                  address = c_index(a_index(k)) + a_index(k) - a_index(j)
                  Kt(address) = Kt(address) + Ke(j, k)
               end if
            end if
         end do
      end do
   end do  
end subroutine Assemble_Kt

! ---------------------------------------------------------------------------------------

! assemble load vector			
subroutine Assemble_Load()						
   double precision :: eNode(4,2)
   double precision :: NodalLoad(8) ! equivalent nodal load
   integer :: i, j, k

   R(:) = 0

   ! assemble load vector for nodal load
   do i=1, node_n
      do j=1, 2
         R(Node(i).eq_n(j)) = Node(i).pm(j)
      end do
   end do
  
   ! assemble load vector for body force
   do i=1, element_n
      ! nodal position of element
      do j=1, 4
        do k=1, 2
           eNode(j, k) = Node( Element(i).cn(j) ).x(k)
        end do
      end do
      ! calculate equivalent nodal load from body force
      call Plane_Load(eNode, Element(i).q, NodalLoad)
      ! assemble load vector
      do j=1, 2
         do k=1, 4
            R( Node( Element(i).cn(k) ).eq_n(j) ) = R( Node( Element(i).cn(k) ).eq_n(j) ) &
                                                  + NodalLoad(4*j+k-4)
         end do   	 
      end do
   end do
end subroutine Assemble_Load

! ---------------------------------------------------------------------------------------

! love linear equations
subroutine Solve_Equations()
   integer :: i

   U(:) = 0.0d0; i=3
   U(1:free_dof_n) = R(1:free_dof_n)
   call COLSOL(Kt(1:c_index(free_dof_n+1)-1), U(1:free_dof_n), c_index(1:free_dof_n+1), &
               free_dof_n,  c_index(free_dof_n+1)-1, free_dof_n+1, 1, i)
   call COLSOL(Kt(1:c_index(free_dof_n+1)-1), U(1:free_dof_n), c_index(1:free_dof_n+1), &
               free_dof_n,  c_index(free_dof_n+1)-1, free_dof_n+1, 2, i) 
end subroutine Solve_Equations

! ---------------------------------------------------------------------------------------

! calculate stress and print solutions
subroutine Displacement_Stress()
   double precision :: eNode(4,2)  ! nodal position of element
   double precision :: displace(8) ! nodal displacement vector of element
   
   double precision :: Stress(4,3) ! Sxx, Syy, Sxy in Gauss points or nodal positions (2*2)
   integer :: i, j, k
   
   ! print strain energy
   write(3,"(a17, E14.6)") "STRAIN ENERGY = ", 0.5d0*dot_product(R(1:free_dof_n), U(1:free_dof_n))
   write(4,*) element_n

   ! print nodal displacement
   write(3,*)
   write(3,*) "DISPLACEMENT "
   write(3,*) "------------"
   write(3,*) "  Node      Dx         Dy     "
   do i=1, node_n
      write(3,"(1x,i4,2x,2(1P,E11.3))") i, U(Node(i).eq_n(1)), U(Node(i).eq_n(2))
   end do
   write(3,*)

   do i=1, element_n
      ! nodal position of element
      do j=1, 4
        do k=1, 2
           eNode(j, k) = Node( Element(i).cn(j) ).x(k)
        end do
      end do
      ! displacement vector of element 
      do j=1, 2
         do k=1, 4
            displace(4*j+k-4) = U( Node( Element(i).cn(k) ).eq_n(j) )
         end do
      end do
      ! calculate stress of element
      call Plane_Stress(Material.Young, Material.Poisson, eNode, displace, Stress)
      ! print stress
      write(3,"(a21,i4)") " STRESS of ELEMENT : ", i
      write(3,*) "------------------------" 
      write(3,*) " Position       Sxx        Syy        Sxy     "
      do j=1, 4
         write(3,"(1x,i4,5x, 3x,3(1P,E11.3))") j, Stress(j,:)
      end do
      write(3,*)
     ! print deformed shape and stress for post processing by using MATLAB 
      write(4,"(1x,28(1P,E13.5))") eNode(1,:), displace(1), displace(5), Stress(1,:),&
                                   eNode(2,:), displace(2), displace(6), Stress(2,:),&
                                   eNode(3,:), displace(3), displace(7), Stress(3,:),&
                                   eNode(4,:), displace(4), displace(8), Stress(4,:)
   end do
end subroutine Displacement_Stress

! ---------------------------------------------------------------------------------------

! print 8x8 matrix
subroutine Print_Matrix(M)
   double precision, intent(in) :: M(:,:)
   integer :: i

   write(2,*) "---------------------------"
   do i=1, 8
      write(2,"(8E12.4)" ) M(i,:)
   end do
   write(2,*)
end subroutine Print_Matrix

! ---------------------------------------------------------------------------------------

subroutine Print_Title(n)
   integer, intent(in) :: n
   
   write(n,*) "*************************************"
   write(n,*) "FINITE ELEMENT ANALYSIS OF STRUCTURES"
   write(n,*) "KAIST OSE532 PSLEE"
   write(n,*) "*************************************"
   write(n,*)
end subroutine Print_Title

! ---------------------------------------------------------------------------------------

end program FEM_edu  ! ------------------------- End of Program -------------------------
