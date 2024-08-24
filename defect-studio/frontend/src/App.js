import React, { useState } from "react";
import Navbar from "./components/Navbar";
import Text2Img from "./components/Text2Img";
import Img2Img from "./components/Img2Img";
import styled from "styled-components";
import Inpaint from "./components/Inpaint";
import RemoveBg from "./components/RemoveBg";

// 스타일링 컴포넌트
const AppContainer = styled.div`
  font-family: Arial, sans-serif;
  color: #333;
  text-align: center;
  background-color: #f8f9fa;
  min-height: 100vh;
`;

function App() {
  const [selectedPage, setSelectedPage] = useState("text2img");

  return (
    <AppContainer>
      <Navbar selectedPage={selectedPage} setSelectedPage={setSelectedPage} />
      {selectedPage === "text2img" && <Text2Img />}
      {selectedPage === "img2img" && <Img2Img />}
      {selectedPage === "inpaint" && <Inpaint />}
      {selectedPage === "removebg" && <RemoveBg />}
    </AppContainer>
  );
}

export default App;
