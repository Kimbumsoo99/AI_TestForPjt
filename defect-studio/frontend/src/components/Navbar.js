import React from "react";
import styled from "styled-components";

// 스타일링 컴포넌트
const Nav = styled.nav`
  background-color: #343a40;
  padding: 10px 20px;
`;

const NavList = styled.ul`
  list-style: none;
  display: flex;
  justify-content: center;
  margin: 0;
  padding: 0;
`;

const NavItem = styled.li.attrs((props) => ({
  // isActive를 DOM으로 전달하지 않기 위해 제거
}))`
  margin: 0 15px;
  color: ${({ isActive }) => (isActive ? "#007BFF" : "#fff")};
  cursor: pointer;
  font-weight: ${({ isActive }) => (isActive ? "bold" : "normal")};

  &:hover {
    color: #007bff;
  }
`;

function Navbar({ selectedPage, setSelectedPage }) {
  return (
    <Nav>
      <NavList>
        <NavItem isActive={selectedPage === "text2img"} onClick={() => setSelectedPage("text2img")}>
          Text-to-Image
        </NavItem>
        <NavItem isActive={selectedPage === "img2img"} onClick={() => setSelectedPage("img2img")}>
          Image-to-Image
        </NavItem>
        <NavItem isActive={selectedPage === "inpaint"} onClick={() => setSelectedPage("inpaint")}>
          Inpainting
        </NavItem>
        <NavItem isActive={selectedPage === "removebg"} onClick={() => setSelectedPage("removebg")}>
          Remove Background
        </NavItem>
        <NavItem isActive={selectedPage === "cleanup"} onClick={() => setSelectedPage("cleanup")}>
          Cleanup
        </NavItem>
        <NavItem isActive={selectedPage === "clip"} onClick={() => setSelectedPage("clip")}>
          Cleanup
        </NavItem>
      </NavList>
    </Nav>
  );
}

export default Navbar;
