import React from "react";
import type { HeadFC, PageProps } from "gatsby";

const NotFoundPage: React.FC<PageProps> = () => {
  return (
    <main style={{ padding: 24 }}>
      <h1>Page not found</h1>
      <p>Sorry, we couldn’t find what you were looking for.</p>
    </main>
  );
};

export default NotFoundPage;

export const Head: HeadFC = () => <title>Not found</title>;


