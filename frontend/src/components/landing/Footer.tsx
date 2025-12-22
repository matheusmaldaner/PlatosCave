// PlatosCave/frontend/src/components/landing/Footer.tsx
import React from 'react';
import { Github, Twitter, Linkedin } from 'lucide-react';
import platosCaveLogo from '../../images/platos-cave-logo.png';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    Product: ['Features', 'Pricing', 'API', 'Integrations'],
    Resources: ['Documentation', 'Blog', 'Research', 'Changelog'],
    Company: ['About', 'Careers', 'Contact', 'Press'],
    Legal: ['Privacy', 'Terms', 'Security'],
  };

  return (
    <footer className="border-t border-gray-100 bg-white/50 backdrop-blur-sm">
      <div className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-5">
          {/* Brand Column */}
          <div className="lg:col-span-2">
            <a href="/" className="inline-block">
              <img src={platosCaveLogo} alt="Plato's Cave" className="h-8" />
            </a>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-text-secondary">
              AI-powered research paper analysis and integrity verification. 
              Built for researchers, by researchers.
            </p>
            <p className="mt-4 text-xs text-text-muted">
              A software created by the FINS group for the University of Florida AI Days Hackathon.
            </p>
          </div>

          {/* Link Columns */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h4 className="mb-4 text-sm font-semibold text-text-primary">{category}</h4>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link}>
                    <a
                      href={`#${link.toLowerCase()}`}
                      className="text-sm text-text-secondary transition-colors duration-200 hover:text-text-primary"
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-gray-100 pt-8 md:flex-row">
          <p className="text-sm text-text-muted">
            © {currentYear} Plato's Cave. All rights reserved.
          </p>
          <div className="flex items-center gap-6">
            {/* Social Icons */}
            <a href="#github" className="text-text-muted transition-colors hover:text-text-primary">
              <Github className="h-5 w-5" />
            </a>
            <a href="#twitter" className="text-text-muted transition-colors hover:text-text-primary">
              <Twitter className="h-5 w-5" />
            </a>
            <a href="#linkedin" className="text-text-muted transition-colors hover:text-text-primary">
              <Linkedin className="h-5 w-5" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
