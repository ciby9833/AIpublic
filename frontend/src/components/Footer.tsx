import { useTranslation } from 'react-i18next';
import './Footer.css';

const Footer = () => {
  const { t } = useTranslation();
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="copyright">
          © {currentYear} JT Cargo. {t('footer.allRightsReserved')}
        </div>
      </div>
    </footer>
  );
};

export default Footer;
